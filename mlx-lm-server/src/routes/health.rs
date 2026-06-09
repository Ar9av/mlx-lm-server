use axum::{extract::State, http::header, response::IntoResponse, Json};
use crate::models::HealthResponse;
use crate::state::AppState;
use serde_json::{json, Value};

pub async fn root() -> Json<Value> {
    Json(json!({ "message": "MLX LM Server is running", "docs": "/health" }))
}

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model_loaded: state.mlx.is_loaded().await,
        current_model: state.mlx.current_model().await,
    })
}

pub async fn status(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "model_loaded": state.mlx.is_loaded().await,
        "current_model": state.mlx.current_model().await,
    }))
}

pub async fn llms_txt() -> impl IntoResponse {
    let body = r#"# MLX LM Server
> OpenAI-compatible LLM inference server for Apple Silicon using MLX

## Inference

### POST /v1/chat/completions
Chat with a loaded model (OpenAI format).
Fields: model, messages[{role,content}], max_tokens, temperature, top_p, stream, kv_bits, kv_group_size, adapter_name
Response: {id, object, created, model, choices[{message,finish_reason}], usage}
Stream: SSE — data: {choices[{delta}]}\n\ndata: [DONE]

### POST /v1/completions
Raw text completion (legacy OpenAI format).
Fields: model, prompt, max_tokens, temperature, top_p, kv_bits, kv_group_size

### POST /v1/embeddings
Generate L2-normalized embeddings from the loaded model's embedding layer.
Fields: model, input (string or string[])
Response: {object:"list", data[{embedding:float[], index}], model, usage}

### POST /v1/messages
Anthropic Messages API compatible endpoint.
Fields: model, messages, system, max_tokens, temperature, stream, kv_bits, kv_group_size, adapter_name
Streaming: SSE Anthropic event sequence (message_start → content_block_delta* → message_stop)

### POST /v1/tokenize
Tokenize a string with the loaded model's tokenizer.
Fields: prompt, model (optional)
Response: {tokens: int[], count: int}

## Model Management

### GET /v1/models
List all locally cached models. Loaded model pinned first.

### POST /v1/models/load
Load a model (downloads if not cached).
Fields: model (HuggingFace ID), adapter (LoRA path, optional), drafter (draft model ID for speculative decoding, optional)

### DELETE /v1/models/{model_id}
Unload currently loaded model and clear MLX cache.

### GET /api/models/local
List locally cached HuggingFace models with size and quantization info.

### DELETE /api/models/local/{org}/{model}
Delete a model from the local HuggingFace cache.

### GET /api/huggingface/models?search=&limit=20
Search mlx-community models on HuggingFace Hub.

### GET /api/ps
Process info: loaded model name, loaded_at timestamp, memory_mb (RSS), pid.

## Adapter Management (LoRA hot-swap)

### GET /v1/adapters
List all mounted adapters with name, base_model, adapter_path, mounted_at.

### POST /v1/adapters/mount
Mount a named LoRA adapter (up to 4 co-resident).
Fields: name, adapter_path, model (optional, defaults to currently loaded)

### DELETE /v1/adapters/{name}
Unmount adapter by name.

Use adapter_name field in chat/messages requests to select a mounted adapter per-request.

## Vision (multimodal)

Attach images to messages using content parts array:
{"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"https://..."}}]}
Or Anthropic format: {"type":"image","source":{"type":"url","url":"..."}}
Requires: pip install mlx-vlm and a vision-capable model.

## Performance options

kv_bits: int (4 or 8) — quantize KV cache for 31-62% decode speedup at long context
kv_group_size: int (default 64) — KV quantization group size
drafter: string — draft model ID for speculative decoding (+1.4x throughput)

## Health

### GET /health
{"status":"ok","model_loaded":bool,"current_model":string|null}

### GET /llms.txt
This document (machine-readable API reference).
"#;
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
}
