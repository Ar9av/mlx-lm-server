use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::env;
use tokio::fs;
use tracing::warn;

use crate::models::{HfModel, LocalModel, ModelList, ModelLoadRequest, ModelObject, QuantizationInfo};
use crate::state::AppState;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let mut data = Vec::new();
    if let Some(id) = state.mlx.current_model().await {
        data.push(ModelObject::new(id));
    }
    Json(ModelList::new(data))
}

pub async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<ModelLoadRequest>,
) -> impl IntoResponse {
    match state.mlx.load_model(req.model.clone()).await {
        Ok(_) => (StatusCode::OK, Json(json!(ModelObject::new(req.model)))).into_response(),
        Err(e) => e.into_response(),
    }
}

pub async fn unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    match state.mlx.current_model().await {
        Some(current) if current == model_id => {
            state.mlx.unload_model().await;
            (StatusCode::OK, Json(json!({ "status": "ok", "deleted": model_id }))).into_response()
        }
        _ => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("Model '{}' not found", model_id),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response(),
    }
}

// ── Local cache ───────────────────────────────────────────────────────────────

fn hf_cache_dir() -> PathBuf {
    env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".cache/huggingface/hub")
}

pub async fn list_local_models() -> Json<Vec<LocalModel>> {
    Json(scan_local_models().await)
}

async fn scan_local_models() -> Vec<LocalModel> {
    let cache = hf_cache_dir();
    let mut models = Vec::new();

    let mut dir = match fs::read_dir(&cache).await {
        Ok(d) => d,
        Err(_) => return models,
    };

    while let Ok(Some(entry)) = dir.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("models--") {
            continue;
        }

        let snapshots = entry.path().join("snapshots");
        let mut snaps = match fs::read_dir(&snapshots).await {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
        while let Ok(Some(snap)) = snaps.next_entry().await {
            if let Ok(meta) = snap.metadata().await {
                if let Ok(modified) = meta.modified() {
                    if latest.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) {
                        latest = Some((modified, snap.path()));
                    }
                }
            }
        }

        if let Some((_, snap_path)) = latest {
            let size_bytes = dir_size(&snap_path).await;
            let quantization = read_quantization(&snap_path).await;
            let model_id = name
                .strip_prefix("models--")
                .unwrap_or(&name)
                .replace("--", "/");

            models.push(LocalModel { id: model_id, size_bytes, quantization });
        }
    }
    models
}

async fn dir_size(path: &PathBuf) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![path.clone()];
    while let Some(p) = stack.pop() {
        let mut rd = match fs::read_dir(&p).await {
            Ok(d) => d,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let meta = match entry.metadata().await {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

async fn read_quantization(snap: &PathBuf) -> Option<QuantizationInfo> {
    let config_path = snap.join("config.json");
    let data = fs::read_to_string(&config_path).await.ok()?;
    let cfg: Value = serde_json::from_str(&data).ok()?;
    let q = cfg.get("quantization").or_else(|| cfg.get("quantization_config"))?;
    Some(QuantizationInfo {
        bits: q.get("bits").cloned(),
        group_size: q.get("group_size").cloned(),
    })
}

pub async fn delete_local_model(
    Path((org, model)): Path<(String, String)>,
) -> impl IntoResponse {
    let dir_name = format!("models--{}--{}", org, model.replace('/', "--"));
    let model_path = hf_cache_dir().join(&dir_name);

    if !model_path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Model not found in cache" })),
        )
            .into_response();
    }

    match tokio::fs::remove_dir_all(&model_path).await {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({ "status": "ok", "deleted": format!("{}/{}", org, model) })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ── HuggingFace search ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct HfSearchQuery {
    #[serde(default)]
    pub search: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    20
}

pub async fn search_hf_models(
    Query(q): Query<HfSearchQuery>,
) -> impl IntoResponse {
    let limit = q.limit.min(50);
    let local_ids: HashSet<String> = scan_local_models()
        .await
        .into_iter()
        .map(|m| m.id)
        .collect();

    let client = reqwest::Client::new();
    let mut params = vec![
        ("author", "mlx-community".to_string()),
        ("sort", "downloads".to_string()),
        ("direction", "-1".to_string()),
        ("limit", limit.to_string()),
        ("full", "true".to_string()),
    ];
    if !q.search.is_empty() {
        params.push(("search", q.search.clone()));
    }

    let resp = match client
        .get("https://huggingface.co/api/models")
        .query(&params)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("HuggingFace API unreachable: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("Failed to reach HuggingFace: {}", e) })),
            )
                .into_response();
        }
    };

    let data: Vec<Value> = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    let mut models: Vec<HfModel> = data
        .iter()
        .map(|item| {
            let id = item["id"].as_str().unwrap_or("").to_string();
            HfModel {
                model_id: id.clone(),
                pipeline_tag: item["pipeline_tag"].as_str().unwrap_or("").to_string(),
                downloads: item["downloads"].as_u64().unwrap_or(0),
                likes: item["likes"].as_u64().unwrap_or(0),
                size_bytes: item["usedStorage"].as_u64(),
                cached: local_ids.contains(&id),
                id,
            }
        })
        .collect();

    // Enrich missing sizes concurrently
    let needs_size: Vec<String> = models
        .iter()
        .filter(|m| m.size_bytes.is_none())
        .map(|m| m.id.clone())
        .collect();

    if !needs_size.is_empty() {
        let client2 = client.clone();
        let size_futures: Vec<_> = needs_size
            .iter()
            .map(|mid| {
                let c = client2.clone();
                let id = mid.clone();
                async move {
                    let url = format!("https://huggingface.co/api/models/{}", id);
                    let resp: Option<Value> = c
                        .get(&url)
                        .timeout(std::time::Duration::from_secs(5))
                        .send()
                        .await
                        .ok()
                        .and_then(|r| futures::executor::block_on(r.json()).ok());
                    let size = resp.as_ref().and_then(|v| v["usedStorage"].as_u64());
                    (id, size)
                }
            })
            .collect();

        let sizes: Vec<(String, Option<u64>)> = futures::future::join_all(size_futures).await;
        let size_map: std::collections::HashMap<String, Option<u64>> = sizes.into_iter().collect();
        for m in &mut models {
            if m.size_bytes.is_none() {
                m.size_bytes = *size_map.get(&m.id).unwrap_or(&None);
            }
        }
    }

    (StatusCode::OK, Json(json!(models))).into_response()
}
