use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;

use crate::models::{AdapterListResponse, AdapterMountRequest};
use crate::state::AppState;

pub async fn list_adapters(State(state): State<AppState>) -> impl IntoResponse {
    let adapters = state.mlx.list_adapters().await;
    Json(AdapterListResponse { object: "list", adapters })
}

pub async fn mount_adapter(
    State(state): State<AppState>,
    Json(req): Json<AdapterMountRequest>,
) -> impl IntoResponse {
    let base_model = match req.model {
        Some(m) => m,
        None => match state.mlx.current_model().await {
            Some(m) => m,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "No model loaded and no model specified in request"})),
                ).into_response()
            }
        },
    };

    match state.mlx.mount_adapter(req.name.clone(), base_model, req.adapter_path).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"status": "mounted", "name": req.name})),
        ).into_response(),
        Err(e) => e.into_response(),
    }
}

pub async fn unmount_adapter(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if state.mlx.unmount_adapter(&name).await {
        (StatusCode::OK, Json(json!({"status": "unmounted", "name": name}))).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Adapter '{}' is not mounted", name)})),
        ).into_response()
    }
}
