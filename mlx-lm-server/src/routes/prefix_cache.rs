use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
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
