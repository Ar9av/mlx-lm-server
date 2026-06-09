use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No TTS model loaded")]
    TtsNotLoaded,

    #[error("No STT model loaded")]
    SttNotLoaded,

    #[error("No STS model loaded")]
    StsNotLoaded,

    #[error("Model load failed: {0}")]
    LoadFailed(String),

    #[error("Input too long: {0} chars (max {1})")]
    InputTooLong(usize, usize),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Audio processing failed: {0}")]
    AudioProcessing(String),

    #[error("Transcription failed: {0}")]
    Transcription(String),

    #[error("Python error: {0}")]
    Python(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AudioError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            AudioError::TtsNotLoaded | AudioError::SttNotLoaded | AudioError::StsNotLoaded => {
                (StatusCode::SERVICE_UNAVAILABLE, "model_not_loaded")
            }
            AudioError::LoadFailed(_) => (StatusCode::BAD_REQUEST, "model_load_error"),
            AudioError::InputTooLong(..) => (StatusCode::BAD_REQUEST, "input_too_long"),
            AudioError::UnsupportedFormat(_) => (StatusCode::BAD_REQUEST, "unsupported_format"),
            AudioError::AudioProcessing(_) | AudioError::Transcription(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "audio_processing_error")
            }
            AudioError::Python(_) | AudioError::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
            }
        };

        (status, Json(json!({
            "error": { "message": self.to_string(), "type": "audio_error", "code": code }
        }))).into_response()
    }
}
