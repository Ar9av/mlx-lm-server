use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use tracing::info;

use crate::mlx_service::MlxService;
use crate::models::{AnthropicRequest, AnthropicResponse, ChatMessage, SamplerParams};
use crate::state::AppState;

pub async fn messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> impl IntoResponse {
    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone(), None).await {
            return e.into_response();
        }
    }

    use crate::models::MessageContent;
    // Convert Anthropic messages → internal ChatMessage format
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut image_urls: Vec<String> = Vec::new();
    if let Some(sys) = &req.system {
        messages.push(ChatMessage { role: "system".into(), content: MessageContent::Text(sys.clone()) });
    }
    for m in &req.messages {
        image_urls.extend(m.content.image_urls());
        messages.push(ChatMessage { role: m.role.clone(), content: MessageContent::Text(m.content.as_text()) });
    }

    // If images present, route to vision handler
    if !image_urls.is_empty() {
        let max_tokens = req.max_tokens;
        let temperature = req.temperature.unwrap_or(state.config.default_temperature);
        let top_p = req.top_p.unwrap_or(state.config.default_top_p);
        let _permit = match state.acquire_inference_slot().await {
            Ok(p) => p,
            Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
        };
        let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
        return match state.mlx.generate_vision_response(messages, image_urls, max_tokens, temperature, top_p).await {
            Ok((content, pt, ct)) => {
                let resp = crate::models::AnthropicResponse::new(
                    crate::mlx_service::MlxService::new_msg_id(), model_name, content, pt, ct,
                );
                (StatusCode::OK, Json(resp)).into_response()
            }
            Err(e) => e.into_response(),
        };
    }

    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    let max_tokens = req.max_tokens;
    let sampler = SamplerParams {
        temperature: req.temperature.unwrap_or(state.config.default_temperature),
        top_p: req.top_p.unwrap_or(state.config.default_top_p),
        ..Default::default()
    };
    let kv_bits = req.kv_bits;
    let kv_group_size = req.kv_group_size;
    let adapter_name = req.adapter_name.clone();

    let _permit = match state.acquire_inference_slot().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    if req.stream {
        stream_messages(state, messages, model_name, max_tokens, sampler, kv_bits, kv_group_size, adapter_name, _permit).await
    } else {
        sync_messages(state, messages, model_name, max_tokens, sampler, kv_bits, kv_group_size, adapter_name, _permit).await
    }
}

async fn sync_messages(
    state: AppState,
    messages: Vec<ChatMessage>,
    model_name: String,
    max_tokens: usize,
    sampler: SamplerParams,
    kv_bits: Option<u32>,
    kv_group_size: Option<u32>,
    adapter_name: Option<String>,
    _permit: tokio::sync::OwnedSemaphorePermit,
) -> axum::response::Response {
    info!("Anthropic sync message for model {}", model_name);
    match state.mlx.generate_response(messages, max_tokens, sampler, serde_json::Value::Object(Default::default()), kv_bits, kv_group_size, adapter_name, None, vec![], false, 0, None, None).await {
        Ok((content, prompt_tokens, completion_tokens, _tool_calls, _finish_reason, _lp, _reasoning)) => {
            let resp = AnthropicResponse::new(
                MlxService::new_msg_id(),
                model_name,
                content,
                prompt_tokens,
                completion_tokens,
            );
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => e.into_response(),
    }
}

async fn stream_messages(
    state: AppState,
    messages: Vec<ChatMessage>,
    model_name: String,
    max_tokens: usize,
    sampler: SamplerParams,
    kv_bits: Option<u32>,
    kv_group_size: Option<u32>,
    adapter_name: Option<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> axum::response::Response {
    let msg_id = MlxService::new_msg_id();
    let timeout = state.config.stream_timeout;

    let (token_stream, _) = match state.mlx.generate_stream(
        msg_id.clone(), messages, max_tokens, sampler, timeout,
        serde_json::Value::Object(Default::default()), kv_bits, kv_group_size, adapter_name, None, vec![], false, 0, None, None,
    ).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };

    let msg_id_clone = msg_id.clone();
    let model_clone = model_name.clone();
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;

    // Anthropic SSE events
    let start_event = format!(
        "event: message_start\ndata: {}\n\n",
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_name,
                "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })
    );
    let block_start = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n";
    let ping = "event: ping\ndata: {\"type\":\"ping\"}\n\n";

    let delta_stream = token_stream.map(move |result| -> Result<Bytes, std::io::Error> {
        let data = match result {
            Ok(token) => {
                output_tokens += 1;
                format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": token}
                    })
                )
            }
            Err(e) => format!(
                "event: error\ndata: {}\n\n",
                serde_json::json!({"type": "error", "error": {"type": "server_error", "message": e.to_string()}})
            ),
        };
        Ok(Bytes::from(data))
    });

    let finish_events = format!(
        "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
         event: message_delta\ndata: {}\n\n\
         event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
        serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        })
    );

    let preamble = futures::stream::iter([
        Ok::<Bytes, std::io::Error>(Bytes::from(start_event)),
        Ok(Bytes::from(block_start)),
        Ok(Bytes::from(ping)),
    ]);

    let combined = preamble
        .chain(delta_stream)
        .chain(futures::stream::once(async move {
            drop(permit);
            Ok::<Bytes, std::io::Error>(Bytes::from(finish_events))
        }));

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));
    (headers, axum::body::Body::from_stream(combined)).into_response()
}
