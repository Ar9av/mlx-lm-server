use axum::{extract::State, http::StatusCode, Json};
use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let (model, quantize, model_path) = state.svc.current_model().await
        .map(|(m, q, p)| (Some(m), q, p))
        .unwrap_or((None, None, None));
    (StatusCode::OK, Json(serde_json::json!({
        "status": "ok",
        "model_loaded": model.is_some(),
        "current_model": model,
        "quantize": quantize,
        "model_path": model_path,
    })))
}
