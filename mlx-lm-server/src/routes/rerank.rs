use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use tracing::info;

use crate::models::{RerankRequest, RerankResponse, RerankResult, Usage};
use crate::state::AppState;

pub async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> impl IntoResponse {
    if req.documents.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "documents array is empty"})),
        )
            .into_response();
    }

    let model_name = req
        .model
        .clone()
        .or_else(|| futures::executor::block_on(state.mlx.current_model()))
        .unwrap_or_else(|| "unknown".into());

    if !state.mlx.is_loaded().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "no model loaded"})),
        )
            .into_response();
    }

    info!("Reranking {} documents against query", req.documents.len());

    let doc_count = req.documents.len();
    let scores = match state.mlx.rerank(req.query, req.documents.clone()).await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };

    let mut results: Vec<RerankResult> = scores
        .into_iter()
        .enumerate()
        .map(|(i, score)| RerankResult {
            index: i,
            document: if req.return_documents {
                req.documents.get(i).cloned()
            } else {
                None
            },
            relevance_score: score,
        })
        .collect();

    // Sort by descending relevance score
    results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top_n limit
    if let Some(n) = req.top_n {
        results.truncate(n);
    }

    let response = RerankResponse {
        object: "list",
        results,
        model: model_name,
        usage: Usage {
            prompt_tokens: doc_count + 1,
            completion_tokens: 0,
            total_tokens: doc_count + 1,
            ..Default::default()
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}
