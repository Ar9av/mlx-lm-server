use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::mlx_service::MlxService;
use crate::models::{ChatMessage, MessageContent, SamplerParams};
use crate::state::AppState;

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<ResponseInputMessage>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponseInputMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateResponseRequest {
    pub model: String,
    pub input: ResponseInput,
    /// System prompt (prepended before conversation history)
    pub instructions: Option<String>,
    /// Resume from a previous response — server loads that conversation and appends new input
    pub previous_response_id: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    pub reasoning_effort: Option<String>,
    pub thinking_budget: Option<usize>,
}

// ── Storage ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StoredResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: String,
    pub output: Vec<OutputItem>,
    pub usage: ResponseUsage,
    /// Full conversation history (input + output), for chaining via previous_response_id
    pub conversation: Vec<ResponseInputMessage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputItem {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: String,
    pub role: &'static str,
    pub content: Vec<OutputContent>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

pub type ResponsesStore = Arc<Mutex<HashMap<String, StoredResponse>>>;

// ── Handlers ──────────────────────────────────────────────────────────────────

pub async fn create_response(
    State(state): State<AppState>,
    Json(req): Json<CreateResponseRequest>,
) -> Response {
    // Build conversation: start from previous response if given, else empty
    // (validate previous_response_id before loading the model)
    let mut conversation: Vec<ResponseInputMessage> = if let Some(ref prev_id) = req.previous_response_id {
        let guard = state.responses.lock().await;
        match guard.get(prev_id) {
            Some(prev) => prev.conversation.clone(),
            None => return (StatusCode::NOT_FOUND, Json(json!({
                "error": {"message": format!("previous_response_id '{}' not found", prev_id), "type": "invalid_request_error"}
            }))).into_response(),
        }
    } else {
        Vec::new()
    };

    // Prepend instructions as system message if provided and not already present
    if let Some(ref instructions) = req.instructions {
        if !conversation.iter().any(|m| m.role == "system") {
            conversation.insert(0, ResponseInputMessage { role: "system".into(), content: instructions.clone() });
        }
    }

    // Append new input
    match req.input {
        ResponseInput::Text(text) => {
            conversation.push(ResponseInputMessage { role: "user".into(), content: text });
        }
        ResponseInput::Messages(msgs) => {
            conversation.extend(msgs);
        }
    }

    let messages: Vec<ChatMessage> = conversation.iter().map(|m| ChatMessage {
        role: m.role.clone(),
        content: MessageContent::Text(m.content.clone()),
    }).collect();

    let max_tokens = req.max_output_tokens.unwrap_or(2048);
    let thinking_budget = req.thinking_budget.or_else(|| {
        req.reasoning_effort.as_deref().map(|e| match e {
            "low" => 512, "high" => 8192, _ => 2048,
        })
    });

    let sampler = SamplerParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        thinking_budget,
        ..Default::default()
    };

    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone(), None).await {
            return e.into_response();
        }
    }

    let _permit = match state.acquire_inference_slot().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "inference queue closed"}))).into_response(),
    };

    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());

    match state.mlx.generate_response(
        messages, max_tokens, sampler,
        serde_json::Value::Null, None, None, None, None,
        vec![], false, 0, None, None,
    ).await {
        Ok((content, prompt_tokens, completion_tokens, _, _, _, _)) => {
            let resp_id = format!("resp_{}", &Uuid::new_v4().to_string().replace('-', "")[..20]);
            let item_id = format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..16]);

            // Append assistant turn to conversation for future chaining
            conversation.push(ResponseInputMessage { role: "assistant".into(), content: content.clone() });

            let stored = StoredResponse {
                id: resp_id.clone(),
                object: "response",
                created_at: now_secs(),
                model: model_name,
                status: "completed".into(),
                output: vec![OutputItem {
                    kind: "message",
                    id: item_id,
                    role: "assistant",
                    content: vec![OutputContent { kind: "output_text", text: content }],
                }],
                usage: ResponseUsage {
                    input_tokens: prompt_tokens,
                    output_tokens: completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
                conversation,
            };

            state.responses.lock().await.insert(resp_id, stored.clone());
            (StatusCode::OK, Json(stored)).into_response()
        }
        Err(e) => e.into_response(),
    }
}

pub async fn get_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    let guard = state.responses.lock().await;
    match guard.get(&response_id) {
        Some(r) => (StatusCode::OK, Json(r.clone())).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({
            "error": {"message": format!("Response '{}' not found", response_id), "type": "invalid_request_error"}
        }))).into_response(),
    }
}

pub async fn delete_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    let mut guard = state.responses.lock().await;
    if guard.remove(&response_id).is_some() {
        Json(json!({"id": response_id, "object": "response.deleted", "deleted": true})).into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(json!({
            "error": {"message": format!("Response '{}' not found", response_id), "type": "invalid_request_error"}
        }))).into_response()
    }
}

pub async fn list_responses(State(state): State<AppState>) -> impl IntoResponse {
    let guard = state.responses.lock().await;
    let data: Vec<&StoredResponse> = guard.values().collect();
    Json(json!({"object": "list", "data": data}))
}
