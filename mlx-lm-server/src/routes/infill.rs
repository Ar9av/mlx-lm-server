use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::models::SamplerParams;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct InfillRequest {
    pub model: Option<String>,
    pub input_prefix: String,
    pub input_suffix: String,
    /// Optional cross-file context snippets for repo-level FIM.
    #[serde(default)]
    pub input_extra: Vec<ExtraContext>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    /// Stop strings to trim from the generated middle section.
    #[serde(default)]
    pub stop: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ExtraContext {
    pub filename: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct InfillResponse {
    pub model: String,
    pub content: String,
    pub usage: InfillUsage,
}

#[derive(Debug, Serialize)]
pub struct InfillUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn infill(State(state): State<AppState>, Json(req): Json<InfillRequest>) -> Response {
    if !state.mlx.is_loaded().await {
        let model_id = match &req.model {
            Some(m) if !m.is_empty() => m.clone(),
            _ => return (StatusCode::BAD_REQUEST, Json(json!({
                "error": {"message": "No model loaded and no model specified", "type": "invalid_request_error"}
            }))).into_response(),
        };
        if let Err(e) = state.mlx.load_model(model_id, None).await {
            return e.into_response();
        }
    }

    let model_name = state.mlx.current_model().await.unwrap_or_default();
    let extra: Vec<(String, String)> = req.input_extra.iter()
        .map(|e| (e.filename.clone(), e.text.clone()))
        .collect();

    let fim_prompt = match state.mlx.build_fim_prompt(
        req.input_prefix.clone(),
        req.input_suffix.clone(),
        extra,
    ).await {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };

    let sampler = SamplerParams {
        temperature: req.temperature.unwrap_or(0.2),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: req.top_k,
        ..Default::default()
    };

    let _permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "inference queue closed"}))).into_response(),
    };

    match state.mlx.generate_completion(fim_prompt, req.max_tokens.unwrap_or(256), sampler, None, None, None).await {
        Ok((mut content, prompt_tokens, completion_tokens)) => {
            // Trim at user stop strings and common FIM end markers
            let end_markers = ["<|fim_end|>", "<fim_middle>", "</s>", "<|endoftext|>", "<EOT>"];
            for stop in req.stop.iter().map(|s| s.as_str()).chain(end_markers.iter().copied()) {
                if let Some(idx) = content.find(stop) {
                    content.truncate(idx);
                }
            }
            (StatusCode::OK, Json(InfillResponse {
                model: model_name,
                content,
                usage: InfillUsage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
            })).into_response()
        }
        Err(e) => e.into_response(),
    }
}
