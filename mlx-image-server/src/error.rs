use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageError {
    #[error("No model loaded — POST /v1/models/load first")]
    NotLoaded,

    #[error("Model load failed: {0}")]
    LoadFailed(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Invalid size: {0}. Use WIDTHxHEIGHT, e.g. 1024x1024")]
    InvalidSize(String),

    #[error("Unsupported model: {0}. Use flux-schnell or flux-dev")]
    UnsupportedModel(String),

    #[error("Python error: {0}")]
    Python(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ImageError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ImageError::NotLoaded => (StatusCode::SERVICE_UNAVAILABLE, "model_not_loaded"),
            ImageError::LoadFailed(_) => (StatusCode::BAD_REQUEST, "model_load_error"),
            ImageError::InvalidSize(_) | ImageError::UnsupportedModel(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            ImageError::GenerationFailed(_) | ImageError::Python(_) | ImageError::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "generation_error")
            }
        };
        (status, Json(json!({
            "error": { "message": self.to_string(), "type": "image_error", "code": code }
        }))).into_response()
    }
}
