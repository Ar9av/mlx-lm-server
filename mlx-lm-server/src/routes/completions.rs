use axum::{extract::{Path, State}, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;
use tracing::info;

use crate::mlx_service::MlxService;
use crate::models::{CompletionRequest, CompletionResponse, SamplerParams, Usage};
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
    let sampler = SamplerParams {
        temperature: req.temperature.unwrap_or(state.config.default_temperature),
        top_p: req.top_p.unwrap_or(state.config.default_top_p),
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        ..Default::default()
    };

    let _permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    info!("Completion request for model {}", model_name);

    let keep_alive = req.keep_alive;
    match state.mlx.generate_completion(
        req.prompt.first().to_string(),
        max_tokens,
        sampler,
        req.kv_bits,
        req.kv_group_size,
        None,
    ).await {
        Ok((text, prompt_tokens, completion_tokens)) => {
            state.mlx.touch_keep_alive(keep_alive);
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

/// DELETE /v1/completions/{request_id} — cancel a running streaming generation.
///
/// The `request_id` is the `id` field returned in the first SSE chunk (e.g. `chatcmpl-xxxx`).
/// Returns 200 if found+cancelled, 404 if the request is not currently active.
pub async fn cancel_completion(
    State(state): State<AppState>,
    Path(request_id): Path<String>,
) -> impl IntoResponse {
    if state.mlx.cancel_request(&request_id) {
        (StatusCode::OK, Json(json!({"cancelled": request_id}))).into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(json!({"error": "request not found or already completed", "id": request_id}))).into_response()
    }
}
