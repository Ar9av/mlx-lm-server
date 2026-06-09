use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MlxError {
    #[error("No model is loaded")]
    NotLoaded,

    #[error("Model load failed: {0}")]
    LoadFailed(String),

    #[error("Model '{0}' is not in the allowed models list")]
    NotAllowed(String),

    #[error("Model '{0}' is {1:.1}GB which exceeds max_model_size_gb ({2:.1}GB)")]
    TooLarge(String, f64, f64),

    #[error("Insufficient RAM: model '{0}' needs ~{1:.1}GB but only {2:.1}GB available")]
    InsufficientRam(String, f64, f64),

    #[error("Adapter '{0}' is not mounted")]
    AdapterNotFound(String),

    #[error("Streaming timed out")]
    Timeout,

    #[error("Token limit exceeded: {0}")]
    TokenLimit(String),

    #[error("Python error: {0}")]
    Python(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for MlxError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            MlxError::NotLoaded => (
                StatusCode::SERVICE_UNAVAILABLE,
                "model_not_loaded",
                self.to_string(),
            ),
            MlxError::LoadFailed(_)
            | MlxError::NotAllowed(_)
            | MlxError::TooLarge(..)
            | MlxError::InsufficientRam(..) => {
                (StatusCode::BAD_REQUEST, "model_load_error", self.to_string())
            }
            MlxError::AdapterNotFound(_) => {
                (StatusCode::NOT_FOUND, "adapter_not_found", self.to_string())
            }
            MlxError::TokenLimit(_) => (
                StatusCode::BAD_REQUEST,
                "message_too_long",
                self.to_string(),
            ),
            MlxError::Timeout => (
                StatusCode::GATEWAY_TIMEOUT,
                "stream_timeout",
                self.to_string(),
            ),
            MlxError::Python(_) | MlxError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_failed",
                self.to_string(),
            ),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": "server_error",
                "code": code,
            }
        });

        (status, Json(body)).into_response()
    }
}
