use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use tracing::info;

use crate::mlx_service::MlxService;
use crate::models::{CompletionRequest, CompletionResponse, Usage};
use crate::state::AppState;

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone(), None).await {
            return e.into_response();
        }
    }

    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    let max_tokens = req.max_tokens.unwrap_or(state.config.default_max_tokens);
    let temperature = req.temperature.unwrap_or(state.config.default_temperature);
    let top_p = req.top_p.unwrap_or(state.config.default_top_p);

    let _permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    info!("Completion request for model {}", model_name);

    match state.mlx.generate_completion(
        req.prompt.first().to_string(),
        max_tokens,
        temperature,
        top_p,
        req.kv_bits,
        req.kv_group_size,
        None,
    ).await {
        Ok((text, prompt_tokens, completion_tokens)) => {
            let resp = CompletionResponse::new(
                MlxService::new_chat_id(),
                model_name,
                text,
                Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
            );
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => e.into_response(),
    }
}
