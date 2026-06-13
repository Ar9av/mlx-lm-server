use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ── TTS ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_speed")]
    pub speed: f64,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    pub language: Option<String>,
    pub ref_audio: Option<String>,
    pub ref_text: Option<String>,
    #[serde(default)]
    pub stream: bool,
}

fn default_voice() -> String { "af_heart".into() }
fn default_speed() -> f64 { 1.0 }
fn default_response_format() -> String { "wav".into() }

// ── STT ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TranscriptionParams {
    pub model: Option<String>,
    pub language: Option<String>,
    #[serde(default = "default_response_format_stt")]
    pub response_format: String,
    pub prompt: Option<String>,
    pub temperature: Option<f64>,
    pub timestamp_granularities: Option<Vec<String>>,
}

fn default_response_format_stt() -> String { "json".into() }

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct TranscriptionSegment {
    pub id: u32,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

// ── STS ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SeparationParams {
    pub model: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SeparationResponse {
    pub target_url: Option<String>,
    pub residual_url: Option<String>,
    pub message: String,
}

// ── Model management ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model: String,
    #[serde(rename = "type")]
    pub model_type: ModelType,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Tts,
    Stt,
    Sts,
    Vad,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Tts => write!(f, "tts"),
            ModelType::Stt => write!(f, "stt"),
            ModelType::Sts => write!(f, "sts"),
            ModelType::Vad => write!(f, "vad"),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
    #[serde(rename = "type")]
    pub model_type: ModelType,
}

impl ModelObject {
    pub fn new(id: String, model_type: ModelType) -> Self {
        Self { id, object: "model", created: now_secs(), owned_by: "mlx-audio-server".into(), model_type }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// ── VAD ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct VadSegment {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Serialize)]
pub struct VadResponse {
    pub model: String,
    pub segments: Vec<VadSegment>,
    pub duration: f64,
}

// ── Health ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub tts_model: Option<String>,
    pub stt_model: Option<String>,
    pub sts_model: Option<String>,
    pub vad_model: Option<String>,
}
