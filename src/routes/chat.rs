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
use crate::models::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent, TokenizeRequest, TokenizeResponse, Usage};
use crate::state::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone(), None).await {
            return e.into_response();
        }
    }

    let total_chars: usize = req.messages.iter().map(|m| m.content.as_text().len()).sum();

    // Check for image content — route to vision handler
    let all_image_urls: Vec<String> = req.messages.iter()
        .flat_map(|m| m.content.image_urls())
        .collect();
    if !all_image_urls.is_empty() {
        let max_tokens = req.max_tokens.unwrap_or(state.config.default_max_tokens);
        let temperature = req.temperature.unwrap_or(state.config.default_temperature);
        let top_p = req.top_p.unwrap_or(state.config.default_top_p);
        let _permit = match state.inference_sem.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
        };
        return match state.mlx.generate_vision_response(
            req.messages.clone(), all_image_urls, max_tokens, temperature, top_p,
        ).await {
            Ok((content, prompt_tokens, completion_tokens)) => {
                let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
                let resp = ChatCompletionResponse::new(
                    MlxService::new_chat_id(),
                    model_name,
                    content,
                    crate::models::Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
                );
                (StatusCode::OK, Json(resp)).into_response()
            }
            Err(e) => e.into_response(),
        };
    }
    if total_chars / 4 > MAX_MESSAGE_TOKENS {
        return MlxError::TokenLimit(format!(
            "estimated {} tokens exceeds maximum {}", total_chars / 4, MAX_MESSAGE_TOKENS
        )).into_response();
    }

    // Inject JSON mode system message if requested
    let mut messages = req.messages.clone();
    if let Some(rf) = &req.response_format {
        if rf.kind == "json_object" {
            let instruction = "You must respond with valid JSON only. Do not include any text before or after the JSON object.".to_string();
            if let Some(first_sys) = messages.iter_mut().find(|m| m.role == "system") {
                let existing = first_sys.content.as_text();
                first_sys.content = MessageContent::Text(format!("{}\n\n{}", existing, instruction));
            } else {
                messages.insert(0, ChatMessage { role: "system".into(), content: MessageContent::Text(instruction) });
            }
        } else if rf.kind == "json_schema" {
            let schema_hint = rf.json_schema.as_ref()
                .and_then(|s| s.get("schema"))
                .map(|s| serde_json::to_string_pretty(s).unwrap_or_default())
                .unwrap_or_default();
            let instruction = format!(
                "Respond only with valid JSON that matches this schema:\n```json\n{}\n```",
                schema_hint
            );
            if let Some(first_sys) = messages.iter_mut().find(|m| m.role == "system") {
                let existing = first_sys.content.as_text();
                first_sys.content = MessageContent::Text(format!("{}\n\n{}", existing, instruction));
            } else {
                messages.insert(0, ChatMessage { role: "system".into(), content: MessageContent::Text(instruction) });
            }
        }
    }

    let max_tokens = req.max_tokens.unwrap_or(state.config.default_max_tokens);
    let temperature = req.temperature.unwrap_or(state.config.default_temperature);
    let top_p = req.top_p.unwrap_or(state.config.default_top_p);
    let chat_template_kwargs = req.chat_template_kwargs.clone();
    let kv_bits = req.kv_bits;
    let kv_group_size = req.kv_group_size;
    let adapter_name = req.adapter_name.clone();

    let permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    if req.stream.unwrap_or(false) {
        stream_response(state, req, messages, max_tokens, temperature, top_p, chat_template_kwargs, kv_bits, kv_group_size, adapter_name, permit).await
    } else {
        sync_response(state, req, messages, max_tokens, temperature, top_p, chat_template_kwargs, kv_bits, kv_group_size, adapter_name, permit).await
    }
}

async fn sync_response(
    state: AppState,
    req: ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    chat_template_kwargs: serde_json::Value,
    kv_bits: Option<u32>,
    kv_group_size: Option<u32>,
    adapter_name: Option<String>,
    _permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    info!("Generating sync response for model {}", model_name);

    match state.mlx.generate_response(messages, max_tokens, temperature, top_p, chat_template_kwargs, kv_bits, kv_group_size, adapter_name).await {
        Ok((content, prompt_tokens, completion_tokens)) => {
            let resp = ChatCompletionResponse::new(
                MlxService::new_chat_id(),
                model_name,
                content,
                Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
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
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    chat_template_kwargs: serde_json::Value,
    kv_bits: Option<u32>,
    kv_group_size: Option<u32>,
    adapter_name: Option<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    let chat_id = MlxService::new_chat_id();
    let timeout = state.config.stream_timeout;

    let token_stream = match state.mlx.generate_stream(
        messages, max_tokens, temperature, top_p, timeout, chat_template_kwargs, kv_bits, kv_group_size, adapter_name,
    ).await {
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
            Err(e) => serde_json::json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            }).to_string(),
        };
        Ok(Bytes::from(format!("data: {}\n\n", data)))
    });

    let final_chunk = ChatCompletionChunk::stop(&chat_id, &model_name);
    let final_data = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        serde_json::to_string(&final_chunk).unwrap_or_default()
    );

    let combined = sse_stream.chain(futures::stream::once(async move {
        drop(permit); // Release semaphore when body is fully consumed
        Ok::<Bytes, std::io::Error>(Bytes::from(final_data))
    }));

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));
    headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));

    (headers, axum::body::Body::from_stream(combined)).into_response()
}

pub async fn tokenize(
    State(state): State<AppState>,
    Json(req): Json<TokenizeRequest>,
) -> impl IntoResponse {
    match state.mlx.tokenize(req.prompt).await {
        Ok(tokens) => {
            let count = tokens.len();
            (StatusCode::OK, Json(TokenizeResponse { tokens, count })).into_response()
        }
        Err(e) => e.into_response(),
    }
}
