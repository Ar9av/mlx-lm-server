use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;

use crate::state::AppState;

pub async fn get_prefix_cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    let cache = state.mlx.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
    let entries: Vec<serde_json::Value> = cache
        .values()
        .map(|e| json!({
            "token_count": e.token_count,
            "hits": e.hits,
            "last_hit": e.last_hit,
        }))
        .collect();
    Json(json!({
        "enabled": state.config.enable_apc,
        "max_entries": state.config.apc_max_entries,
        "entry_count": entries.len(),
        "entries": entries,
    }))
}

pub async fn clear_prefix_cache(State(state): State<AppState>) -> Response {
    let mut cache = state.mlx.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
    let n = cache.len();
    cache.clear();
    (StatusCode::OK, Json(json!({"cleared": n}))).into_response()
}

#[derive(Deserialize, Default)]
pub struct PathQuery {
    pub path: Option<String>,
}

/// POST /v1/prefix_cache/save — pickle all APC entries to disk.
pub async fn save_prefix_cache(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<PathQuery>,
) -> Response {
    let path = q.path
        .or_else(|| state.config.kv_persist_path.clone())
        .unwrap_or_else(|| "/tmp/mlx-lm-prefix-cache.pkl".to_string());
    match state.mlx.save_prefix_cache_to_file(path.clone()).await {
        Ok(n) => (StatusCode::OK, Json(json!({"saved": n, "path": path}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

/// POST /v1/prefix_cache/load — restore APC entries from disk.
pub async fn load_prefix_cache(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<PathQuery>,
) -> Response {
    let path = q.path
        .or_else(|| state.config.kv_persist_path.clone())
        .unwrap_or_else(|| "/tmp/mlx-lm-prefix-cache.pkl".to_string());
    if !std::path::Path::new(&path).exists() {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "file not found", "path": path}))).into_response();
    }
    match state.mlx.load_prefix_cache_from_file(path.clone()).await {
        Ok(n) => (StatusCode::OK, Json(json!({"loaded": n, "path": path}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

/// DELETE /v1/prefix_cache/saved — erase the persisted file.
pub async fn erase_prefix_cache_file(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<PathQuery>,
) -> Response {
    let path = q.path
        .or_else(|| state.config.kv_persist_path.clone())
        .unwrap_or_else(|| "/tmp/mlx-lm-prefix-cache.pkl".to_string());
    match std::fs::remove_file(&path) {
        Ok(()) => (StatusCode::OK, Json(json!({"erased": path}))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({"error": e.to_string(), "path": path}))).into_response(),
    }
}
