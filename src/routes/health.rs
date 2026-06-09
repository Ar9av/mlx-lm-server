use axum::{extract::State, Json};
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
