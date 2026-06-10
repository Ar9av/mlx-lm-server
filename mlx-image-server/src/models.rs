use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    // Extra params beyond OpenAI spec
    pub steps: Option<u32>,
    pub guidance: Option<f64>,
    pub seed: Option<i64>,
    pub negative_prompt: Option<String>,
    pub quantize: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
    pub quantize: Option<u32>,
    /// Override the HuggingFace repo to download weights from.
    /// Useful for public pre-quantized community models, e.g.
    /// "madroid/flux.1-schnell-mflux-4bit"
    pub model_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub loaded: bool,
    pub quantize: Option<u32>,
}
