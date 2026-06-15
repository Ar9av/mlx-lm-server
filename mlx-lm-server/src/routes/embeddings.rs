use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::models::{EmbeddingObject, EmbeddingRequest, EmbeddingResponse, StringOrArray, Usage};
use crate::state::AppState;

pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone(), None).await {
            return e.into_response();
        }
    }

    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    let texts: Vec<String> = match &req.input {
        StringOrArray::Single(s) => vec![s.clone()],
        StringOrArray::Multiple(v) => v.clone(),
    };
    let n = texts.len();

    let _permit = match state.acquire_inference_slot().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    match state.mlx.get_embeddings(texts).await {
        Ok(vecs) => {
            let data = vecs
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| EmbeddingObject {
                    object: "embedding",
                    embedding,
                    index: i as u32,
                })
                .collect();
            let resp = EmbeddingResponse {
                object: "list",
                data,
                model: model_name,
                usage: Usage { prompt_tokens: n, completion_tokens: 0, total_tokens: n, ..Default::default() },
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => e.into_response(),
    }
}
