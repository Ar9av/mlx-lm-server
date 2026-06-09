use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use tracing::{error, info};

use crate::error::MlxError;
use crate::mlx_service::{MlxService, MAX_MESSAGE_TOKENS};
use crate::models::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Usage};
use crate::state::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Auto-load model if not loaded
    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone()).await {
            return e.into_response();
        }
    }

    let messages = req.messages.clone();
    let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
    if total_chars / 4 > MAX_MESSAGE_TOKENS {
        return MlxError::TokenLimit(format!(
            "estimated {} tokens exceeds maximum {}",
            total_chars / 4,
            MAX_MESSAGE_TOKENS
        ))
        .into_response();
    }

    let max_tokens = req.max_tokens.unwrap_or(state.config.default_max_tokens);
    let temperature = req.temperature.unwrap_or(state.config.default_temperature);
    let top_p = req.top_p.unwrap_or(state.config.default_top_p);
    let chat_template_kwargs = req.chat_template_kwargs.clone();

    if req.stream.unwrap_or(false) {
        stream_response(state, req, max_tokens, temperature, top_p, chat_template_kwargs).await
    } else {
        sync_response(state, req, max_tokens, temperature, top_p, chat_template_kwargs).await
    }
}

async fn sync_response(
    state: AppState,
    req: ChatCompletionRequest,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    chat_template_kwargs: serde_json::Value,
) -> Response {
    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    info!("Generating sync response for model {}", model_name);

    match state
        .mlx
        .generate_response(req.messages, max_tokens, temperature, top_p, chat_template_kwargs)
        .await
    {
        Ok((content, prompt_tokens, completion_tokens)) => {
            let resp = ChatCompletionResponse::new(
                MlxService::new_chat_id(),
                model_name,
                content,
                Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            );
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            error!("Generation failed: {}", e);
            e.into_response()
        }
    }
}

async fn stream_response(
    state: AppState,
    req: ChatCompletionRequest,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    chat_template_kwargs: serde_json::Value,
) -> Response {
    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    let chat_id = MlxService::new_chat_id();
    let timeout = state.config.stream_timeout;

    let token_stream = match state
        .mlx
        .generate_stream(req.messages, max_tokens, temperature, top_p, timeout, chat_template_kwargs)
        .await
    {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };

    let chat_id_clone = chat_id.clone();
    let model_clone = model_name.clone();
    let mut first = true;

    let sse_stream = token_stream.map(move |result| -> Result<Bytes, std::io::Error> {
        let data = match result {
            Ok(token) => {
                let chunk = ChatCompletionChunk::token(&chat_id_clone, &model_clone, &token, first);
                first = false;
                serde_json::to_string(&chunk).unwrap_or_default()
            }
            Err(e) => {
                let err_json = serde_json::json!({
                    "error": { "message": e.to_string(), "type": "server_error" }
                });
                serde_json::to_string(&err_json).unwrap_or_default()
            }
        };
        Ok(Bytes::from(format!("data: {}\n\n", data)))
    });

    let final_chunk = ChatCompletionChunk::stop(&chat_id, &model_name);
    let final_data = format!("data: {}\n\ndata: [DONE]\n\n", serde_json::to_string(&final_chunk).unwrap_or_default());

    let combined = sse_stream.chain(futures::stream::once(async move {
        Ok::<Bytes, std::io::Error>(Bytes::from(final_data))
    }));

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));
    headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));

    let body = axum::body::Body::from_stream(combined);
    (headers, body).into_response()
}
