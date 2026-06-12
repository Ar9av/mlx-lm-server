use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::state::AppState;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub struct VectorStoreEntry {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub documents: Vec<StoredDocument>,
}

pub struct StoredDocument {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct CreateVectorStoreRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AddDocumentRequest {
    pub id: Option<String>,
    pub text: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct VectorSearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
    pub score_threshold: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct VectorStoreObject {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub document_count: usize,
}

#[derive(Debug, Serialize)]
pub struct VectorSearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct VectorSearchResponse {
    pub object: &'static str,
    pub results: Vec<VectorSearchResult>,
    pub query: String,
    pub store_id: String,
}

pub type VectorStoreMap = Arc<Mutex<HashMap<String, VectorStoreEntry>>>;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

pub async fn create_vector_store(
    State(state): State<AppState>,
    Json(req): Json<CreateVectorStoreRequest>,
) -> Response {
    let id = Uuid::new_v4().to_string();
    let entry = VectorStoreEntry {
        id: id.clone(),
        name: req.name.clone(),
        created_at: now_secs(),
        documents: Vec::new(),
    };
    let obj = VectorStoreObject {
        id: id.clone(),
        name: req.name,
        created_at: entry.created_at,
        document_count: 0,
    };
    state.vector_stores.lock().await.insert(id, entry);
    (StatusCode::CREATED, Json(obj)).into_response()
}

pub async fn list_vector_stores(State(state): State<AppState>) -> impl IntoResponse {
    let stores = state.vector_stores.lock().await;
    let data: Vec<VectorStoreObject> = stores
        .values()
        .map(|e| VectorStoreObject {
            id: e.id.clone(),
            name: e.name.clone(),
            created_at: e.created_at,
            document_count: e.documents.len(),
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

pub async fn get_vector_store(
    State(state): State<AppState>,
    Path(store_id): Path<String>,
) -> Response {
    let stores = state.vector_stores.lock().await;
    match stores.get(&store_id) {
        Some(e) => {
            let obj = VectorStoreObject {
                id: e.id.clone(),
                name: e.name.clone(),
                created_at: e.created_at,
                document_count: e.documents.len(),
            };
            (StatusCode::OK, Json(obj)).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "vector store not found", "code": "not_found" } })),
        )
            .into_response(),
    }
}

pub async fn delete_vector_store(
    State(state): State<AppState>,
    Path(store_id): Path<String>,
) -> Response {
    let mut stores = state.vector_stores.lock().await;
    if stores.remove(&store_id).is_some() {
        Json(json!({ "id": store_id, "deleted": true })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "vector store not found", "code": "not_found" } })),
        )
            .into_response()
    }
}

pub async fn add_document(
    State(state): State<AppState>,
    Path(store_id): Path<String>,
    Json(req): Json<AddDocumentRequest>,
) -> Response {
    {
        let stores = state.vector_stores.lock().await;
        if !stores.contains_key(&store_id) {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": { "message": "vector store not found", "code": "not_found" } })),
            )
                .into_response();
        }
    }

    let embedding = match state.mlx.get_embeddings(vec![req.text.clone()]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": { "message": "embedding returned no vectors", "code": "embedding_failed" } })),
            )
                .into_response();
        }
        Err(e) => return e.into_response(),
    };

    let doc_id = req.id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let metadata = req.metadata.unwrap_or(serde_json::Value::Null);

    let doc = StoredDocument {
        id: doc_id.clone(),
        text: req.text.clone(),
        embedding,
        metadata: metadata.clone(),
    };

    let mut stores = state.vector_stores.lock().await;
    match stores.get_mut(&store_id) {
        Some(entry) => {
            entry.documents.push(doc);
            (
                StatusCode::CREATED,
                Json(json!({ "id": doc_id, "text": req.text, "metadata": metadata })),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "vector store not found", "code": "not_found" } })),
        )
            .into_response(),
    }
}

pub async fn search_documents(
    State(state): State<AppState>,
    Path(store_id): Path<String>,
    Json(req): Json<VectorSearchRequest>,
) -> Response {
    let docs_snapshot: Vec<(String, String, Vec<f32>, serde_json::Value)> = {
        let stores = state.vector_stores.lock().await;
        match stores.get(&store_id) {
            Some(entry) => {
                if entry.documents.is_empty() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({ "error": { "message": "vector store has no documents", "code": "empty_store" } })),
                    )
                        .into_response();
                }
                entry
                    .documents
                    .iter()
                    .map(|d| (d.id.clone(), d.text.clone(), d.embedding.clone(), d.metadata.clone()))
                    .collect()
            }
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": { "message": "vector store not found", "code": "not_found" } })),
                )
                    .into_response();
            }
        }
    };

    let query_embedding = match state.mlx.get_embeddings(vec![req.query.clone()]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": { "message": "embedding returned no vectors", "code": "embedding_failed" } })),
            )
                .into_response();
        }
        Err(e) => return e.into_response(),
    };

    let top_k = req.top_k.unwrap_or(5);

    let mut scored: Vec<VectorSearchResult> = docs_snapshot
        .into_iter()
        .map(|(id, text, emb, metadata)| {
            let score = cosine_similarity(&query_embedding, &emb);
            VectorSearchResult { id, text, score, metadata }
        })
        .collect();

    if let Some(threshold) = req.score_threshold {
        scored.retain(|r| r.score >= threshold);
    }

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    let resp = VectorSearchResponse {
        object: "vector_search_results",
        results: scored,
        query: req.query,
        store_id,
    };
    (StatusCode::OK, Json(resp)).into_response()
}
