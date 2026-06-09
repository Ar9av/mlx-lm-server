use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Shared ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Default)]
pub struct ResponseFormat {
    #[serde(rename = "type", default)]
    pub kind: String,
    pub json_schema: Option<serde_json::Value>,
}

// ── Chat completions ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub stop: Option<StopSequence>,
    pub n: Option<u32>,
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub chat_template_kwargs: serde_json::Value,
    pub kv_bits: Option<u32>,
    pub kv_group_size: Option<u32>,
    pub adapter_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(id: String, model: String, content: String, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion",
            created: now_secs(),
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage { role: "assistant".into(), content },
                finish_reason: Some("stop".into()),
            }],
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    pub fn token(id: &str, model: &str, content: &str, first: bool) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: now_secs(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: if first { Some("assistant".into()) } else { None }, content: Some(content.to_string()) },
                finish_reason: None,
            }],
        }
    }

    pub fn stop(id: &str, model: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: now_secs(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: None },
                finish_reason: Some("stop".into()),
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Text completions (legacy) ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: StringOrArray,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,
    pub stop: Option<StopSequence>,
    pub suffix: Option<String>,
    pub n: Option<u32>,
    pub kv_bits: Option<u32>,
    pub kv_group_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Multiple(Vec<String>),
}

impl StringOrArray {
    pub fn first(&self) -> &str {
        match self {
            StringOrArray::Single(s) => s.as_str(),
            StringOrArray::Multiple(v) => v.first().map(|s| s.as_str()).unwrap_or(""),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(id: String, model: String, text: String, usage: Usage) -> Self {
        Self {
            id,
            object: "text_completion",
            created: now_secs(),
            model,
            choices: vec![CompletionChoice { text, index: 0, finish_reason: Some("stop".into()) }],
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

// ── Tokenize ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub model: Option<String>,
    pub prompt: String,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i64>,
    pub count: usize,
}

// ── Embeddings ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: StringOrArray,
    pub encoding_format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

// ── Anthropic Messages API ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub system: Option<String>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    pub stop_sequences: Option<Vec<String>>,
    pub kv_bits: Option<u32>,
    pub kv_group_size: Option<u32>,
    pub adapter_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl AnthropicContent {
    pub fn as_text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks.iter()
                .filter_map(|b| if b.kind == "text" { Some(b.text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

impl AnthropicResponse {
    pub fn new(id: String, model: String, text: String, input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            id,
            kind: "message",
            role: "assistant",
            content: vec![ContentBlock { kind: "text".into(), text }],
            model,
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage { input_tokens, output_tokens },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

// ── Model management ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model: String,
    pub adapter: Option<String>,
    pub drafter: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

impl ModelObject {
    pub fn new(id: String) -> Self {
        Self { id, object: "model", created: now_secs(), owned_by: "mlx-lm-server".into() }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

impl ModelList {
    pub fn new(data: Vec<ModelObject>) -> Self {
        Self { object: "list", data }
    }
}

// ── Health / ps ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub model_loaded: bool,
    pub current_model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PsResponse {
    pub model: Option<String>,
    pub loaded_at: Option<u64>,
    pub memory_mb: f64,
    pub pid: u32,
}

// ── Local model cache ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct LocalModel {
    pub id: String,
    pub size_bytes: u64,
    pub quantization: Option<QuantizationInfo>,
}

#[derive(Debug, Serialize)]
pub struct QuantizationInfo {
    pub bits: Option<serde_json::Value>,
    pub group_size: Option<serde_json::Value>,
}

// ── Adapter management ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AdapterMountRequest {
    pub name: String,
    pub adapter_path: String,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MountedAdapterInfo {
    pub name: String,
    pub base_model: String,
    pub adapter_path: String,
    pub mounted_at: u64,
}

#[derive(Debug, Serialize)]
pub struct AdapterListResponse {
    pub object: &'static str,
    pub adapters: Vec<MountedAdapterInfo>,
}

// ── HuggingFace search ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HfModel {
    pub id: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub pipeline_tag: String,
    pub downloads: u64,
    pub likes: u64,
    pub size_bytes: Option<u64>,
    pub cached: bool,
}
