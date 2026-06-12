use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn now_secs_pub() -> u64 { now_secs() }

// ── Shared ────────────────────────────────────────────────────────────────────

/// Per-part content for multimodal messages
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub image_url: Option<ImageUrlRef>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrlRef {
    pub url: String,
}

/// Either a plain text string or an array of typed content parts
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts.iter()
                .filter_map(|p| if p.kind == "text" { p.text.as_deref() } else { None })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    pub fn image_urls(&self) -> Vec<String> {
        match self {
            MessageContent::Text(_) => vec![],
            MessageContent::Parts(parts) => parts.iter()
                .filter(|p| p.kind == "image_url")
                .filter_map(|p| p.image_url.as_ref().map(|u| u.url.clone()))
                .collect(),
        }
    }

    pub fn has_images(&self) -> bool {
        !self.image_urls().is_empty()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Sampling parameters threaded through all generation calls.
#[derive(Debug, Clone, Default)]
pub struct SamplerParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub num_draft_tokens: Option<usize>,
    /// XTC (Exclude Top Choices) sampler — improves diversity by removing highly
    /// probable tokens with probability `xtc_probability` when they exceed `xtc_threshold`.
    pub xtc_probability: Option<f64>,
    pub xtc_threshold: Option<f64>,
    /// Max thinking tokens for reasoning models (Qwen3/DeepSeek-R1).
    /// Forces </think> after this many tokens in the reasoning block.
    pub thinking_budget: Option<usize>,
    /// Map of token_id -> bias (-100 to 100). Positive biases increase likelihood,
    /// negative decrease it. -100 effectively bans the token.
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
    /// EBNF grammar string for grammar-constrained generation via outlines.
    /// Overrides response_format if both are set.
    pub grammar: Option<String>,
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

// ── Tool use ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,  // "function"
    pub function: ToolFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,  // JSON string (OpenAI format)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

impl ToolCall {
    pub fn new(name: String, arguments: serde_json::Value) -> Self {
        Self {
            id: format!("call_{}", &uuid::Uuid::new_v4().to_string()[..8]),
            kind: "function".into(),
            function: ToolCallFunction {
                name,
                arguments: arguments.to_string(),
            },
        }
    }
}

// ── Chat completions ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub stream: Option<bool>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub num_draft_tokens: Option<usize>,
    pub stop: Option<StopSequence>,
    pub n: Option<u32>,
    pub response_format: Option<ResponseFormat>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<serde_json::Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub seed: Option<u64>,
    pub session_id: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: serde_json::Value,
    pub kv_bits: Option<u32>,
    pub kv_group_size: Option<u32>,
    pub adapter_name: Option<String>,
    pub xtc_probability: Option<f64>,
    pub xtc_threshold: Option<f64>,
    /// Max thinking tokens for reasoning models. Forces </think> after N tokens.
    pub thinking_budget: Option<usize>,
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
    pub grammar: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize, Clone)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LogprobsInfo {
    pub content: Vec<TokenLogprob>,
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
                message: ChatMessage { role: "assistant".into(), content: MessageContent::Text(content) },
                finish_reason: Some("stop".into()),
                tool_calls: None,
                logprobs: None,
                reasoning_content: None,
            }],
            usage,
        }
    }

    pub fn with_tool_calls(id: String, model: String, tool_calls: Vec<ToolCall>, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion",
            created: now_secs(),
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage { role: "assistant".into(), content: MessageContent::Text(String::new()) },
                finish_reason: Some("tool_calls".into()),
                tool_calls: Some(tool_calls),
                logprobs: None,
                reasoning_content: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobsInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
        Self::token_lp(id, model, content, first, None)
    }

    pub fn token_lp(id: &str, model: &str, content: &str, first: bool, logprobs: Option<LogprobsInfo>) -> Self {
        Self::token_lp_reasoning(id, model, content, first, logprobs, false)
    }

    pub fn token_lp_reasoning(
        id: &str,
        model: &str,
        content: &str,
        first: bool,
        logprobs: Option<LogprobsInfo>,
        is_reasoning: bool,
    ) -> Self {
        let (content_field, reasoning_field) = if is_reasoning {
            (None, Some(content.to_string()))
        } else {
            (Some(content.to_string()), None)
        };
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: now_secs(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: if first { Some("assistant".into()) } else { None },
                    content: content_field,
                    reasoning_content: reasoning_field,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs,
            }],
        }
    }

    pub fn stop(id: &str, model: &str) -> Self {
        Self::finish(id, model, "stop")
    }

    pub fn finish(id: &str, model: &str, reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: now_secs(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: None, reasoning_content: None, tool_calls: None },
                finish_reason: Some(reason.to_string()),
                logprobs: None,
            }],
        }
    }

    pub fn tool_calls_chunk(id: &str, model: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: now_secs(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant".into()), content: None, reasoning_content: None, tool_calls: Some(tool_calls) },
                finish_reason: Some("tool_calls".to_string()),
                logprobs: None,
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobsInfo>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── Text completions (legacy) ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: StringOrArray,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub stream: Option<bool>,
    pub repetition_penalty: Option<f64>,
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
                .filter_map(|b| if b.kind == "text" { b.text.as_deref() } else { None })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    pub fn image_urls(&self) -> Vec<String> {
        match self {
            AnthropicContent::Text(_) => vec![],
            AnthropicContent::Blocks(blocks) => blocks.iter()
                .filter(|b| b.kind == "image")
                .filter_map(|b| b.source.as_ref())
                .filter_map(|s| s.url.clone())
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentBlockSource {
    #[serde(rename = "type")]
    pub kind: String,
    pub url: Option<String>,
    pub data: Option<String>,
    pub media_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub source: Option<ContentBlockSource>,
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
            content: vec![ContentBlock { kind: "text".into(), text: Some(text), source: None }],
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

// ── Benchmark ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BenchmarkRequest {
    #[serde(default = "default_bench_prompt")]
    pub prompt: String,
    #[serde(default = "default_bench_runs")]
    pub runs: usize,
    #[serde(default = "default_bench_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f64>,
}

fn default_bench_prompt() -> String { "Write a short poem about the ocean.".into() }
fn default_bench_runs() -> usize { 3 }
fn default_bench_max_tokens() -> usize { 64 }

#[derive(Debug, Serialize)]
pub struct BenchmarkResult {
    pub model: Option<String>,
    pub runs: usize,
    pub max_tokens: usize,
    pub ttft_ms_p50: f64,
    pub ttft_ms_p95: f64,
    pub tokens_per_sec_mean: f64,
    pub tokens_per_sec_p50: f64,
    pub total_tokens_generated: usize,
}

// ── Rich model info ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
    pub size_bytes: Option<u64>,
    pub size_gb: Option<f64>,
    pub quantization: Option<QuantizationInfo>,
    pub vision: bool,
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub loaded: bool,
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

// ── Fine-tuning (LoRA / DoRA / full) ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    /// Base model (HuggingFace repo or local path)
    pub model: String,
    /// Directory containing train.jsonl / valid.jsonl / test.jsonl
    pub data: String,
    /// "lora" | "dora" | "full"  (default: "lora")
    #[serde(default = "default_fine_tune_type")]
    pub fine_tune_type: String,
    /// Where to save adapter weights (default: "./adapters")
    #[serde(default = "default_adapter_path")]
    pub adapter_path: String,
    /// Number of transformer layers to fine-tune (default: 16, -1 = all)
    pub num_layers: Option<i32>,
    /// Training iterations (default: 100)
    pub iters: Option<usize>,
    /// Mini-batch size (default: 4)
    pub batch_size: Option<usize>,
    /// Adam learning rate (default: 1e-4)
    pub learning_rate: Option<f64>,
    /// "adam" | "adamw" | "muon" | "sgd" | "adafactor" (default: "adamw")
    pub optimizer: Option<String>,
    /// Max sequence length (default: 2048)
    pub max_seq_length: Option<usize>,
    /// Gradient accumulation steps (default: 1)
    pub grad_accumulation_steps: Option<usize>,
    /// Report every N steps (default: 10)
    pub steps_per_report: Option<usize>,
    /// Validate every N steps (default: 200)
    pub steps_per_eval: Option<usize>,
    /// Save checkpoint every N steps (default: 100)
    pub save_every: Option<usize>,
    /// Mask prompt tokens in loss (default: false)
    #[serde(default)]
    pub mask_prompt: bool,
    /// Resume from existing adapter checkpoint
    pub resume_adapter_file: Option<String>,
    /// Run evaluation on test set after training (default: false)
    #[serde(default)]
    pub test: bool,
    /// Gradient checkpointing to save memory (default: false)
    #[serde(default)]
    pub grad_checkpoint: bool,
}

fn default_fine_tune_type() -> String { "lora".into() }
fn default_adapter_path() -> String { "./adapters".into() }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainProgress {
    pub event: &'static str,   // "progress" | "done" | "error"
    pub step: Option<usize>,
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub tokens_per_sec: Option<f64>,
    pub adapter_path: Option<String>,
    pub message: Option<String>,
}

// ── Fuse (merge adapter into base model) ─────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FuseRequest {
    /// Adapter name (must be mounted) or raw adapter path
    pub adapter: String,
    /// Where to save the fused model (default: ./fused-<adapter>)
    pub output: Option<String>,
    /// Also export as GGUF  (default: false)
    #[serde(default)]
    pub export_gguf: bool,
    /// De-quantize before fusing (default: false)
    #[serde(default)]
    pub dequantize: bool,
    /// Upload fused model to HuggingFace (optional)
    pub upload_repo: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FuseResponse {
    pub output_path: String,
    pub gguf_path: Option<String>,
    pub uploaded_to: Option<String>,
}

// ── Convert (GGUF / HF → MLX) ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ConvertRequest {
    /// Source HuggingFace repo or local path
    pub model: String,
    /// Output path for the converted MLX model (default: ./mlx-<model>)
    pub output: Option<String>,
    /// Quantize bits: 4 or 8 (omit for fp16)
    pub quantize_bits: Option<u32>,
    /// Quantize group size (default: 64)
    pub quantize_group_size: Option<u32>,
    /// Upload to HuggingFace after conversion
    pub upload_repo: Option<String>,
    /// HuggingFace token for private models / uploads
    pub hf_token: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConvertResponse {
    pub output_path: String,
    pub uploaded_to: Option<String>,
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

// ── Pipeline ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct PipelineStep {
    pub name: Option<String>,
    pub system: Option<String>,
    pub user: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct PipelineRequest {
    pub steps: Vec<PipelineStep>,
    pub input: String,
    pub model: Option<String>,
    pub max_tokens_per_step: Option<usize>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct PipelineStepResult {
    pub name: Option<String>,
    pub output: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct PipelineResponse {
    pub steps: Vec<PipelineStepResult>,
    pub output: String,
    pub model: String,
    pub total_prompt_tokens: usize,
    pub total_completion_tokens: usize,
}

// ── Rerank ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    /// Return only the top_n results (default: return all, sorted by score)
    pub top_n: Option<usize>,
    /// Include document text in the response (default: false)
    #[serde(default)]
    pub return_documents: bool,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
    pub relevance_score: f32,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub object: &'static str,
    pub results: Vec<RerankResult>,
    pub model: String,
    pub usage: Usage,
}

// ── Training job ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingJob {
    pub id: String,
    pub object: &'static str,
    pub status: String, // "pending" | "running" | "completed" | "failed" | "cancelled"
    pub model: String,
    pub fine_tune_type: String,
    pub adapter_path: String,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    pub events: Vec<TrainProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ── Batch API ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRequestItem {
    pub custom_id: String,
    pub method: String,
    pub url: String,
    pub body: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    pub endpoint: String,
    pub requests: Vec<BatchRequestItem>,
    pub completion_window: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BatchRequestCounts {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BatchJob {
    pub id: String,
    pub object: &'static str,
    pub endpoint: String,
    pub status: String,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_progress_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<u64>,
    pub request_counts: BatchRequestCounts,
    pub requests: Vec<serde_json::Value>,
    pub results: Vec<serde_json::Value>,
    pub errors: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}
