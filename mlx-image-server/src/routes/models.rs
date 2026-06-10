use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use crate::models::{LoadModelRequest, ModelInfo};
use crate::state::AppState;

pub async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> impl IntoResponse {
    let quantize = req.quantize.or(state.config.default_quantize);
    match state.svc.load(req.model, quantize, req.model_path).await {
        Ok(()) => {
            let (id, q, mp) = state.svc.current_model().await.unwrap_or_default();
            (StatusCode::OK, Json(serde_json::json!({
                "status": "loaded",
                "model": id,
                "quantize": q,
                "model_path": mp,
            }))).into_response()
        }
        Err(e) => e.into_response(),
    }
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let models: Vec<ModelInfo> = if let Some((id, q, _)) = state.svc.current_model().await {
        vec![ModelInfo { id, object: "model", loaded: true, quantize: q }]
    } else {
        vec![]
    };
    (StatusCode::OK, Json(serde_json::json!({ "object": "list", "data": models }))).into_response()
}

pub async fn unload_model(State(state): State<AppState>) -> impl IntoResponse {
    state.svc.unload().await;
    (StatusCode::OK, Json(serde_json::json!({ "status": "unloaded" }))).into_response()
}
