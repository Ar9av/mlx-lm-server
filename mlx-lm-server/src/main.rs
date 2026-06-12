mod config;
mod error;
mod mlx_service;
mod models;
mod routes;
mod state;

use axum::{
    extract::Request,
    http::{HeaderName, HeaderValue},
    middleware::{self, Next},
    response::Response,
    routing::{delete, get, post},
    Router,
};
use state::AppState;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use uuid::Uuid;

static X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

async fn request_id_middleware(req: Request, next: Next) -> Response {
    let id = req
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let mut resp = next.run(req).await;
    if let Ok(val) = HeaderValue::from_str(&id) {
        resp.headers_mut().insert(&X_REQUEST_ID, val);
    }
    resp
}

#[tokio::main]
async fn main() {
    let cfg = config::Config::from_env();

    let level = if cfg.debug { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level)))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let state = AppState::new(cfg.clone());

    let cors = CorsLayer::new()
        .allow_origin(
            cfg.cors_origins
                .iter()
                .map(|o| o.parse().expect("invalid CORS origin"))
                .collect::<Vec<_>>(),
        )
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        // Info
        .route("/", get(routes::health::root))
        .route("/health", get(routes::health::health))
        .route("/status", get(routes::health::status))
        .route("/llms.txt", get(routes::health::llms_txt))
        // OpenAI-compatible
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/tokenize", post(routes::chat::tokenize))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/models/load", post(routes::models::load_model))
        .route("/v1/models/:model_id/info", get(routes::models::model_info))
        .route("/v1/models/:model_id", delete(routes::models::unload_model))
        // Adapter management
        .route("/v1/adapters", get(routes::adapters::list_adapters))
        .route("/v1/adapters/mount", post(routes::adapters::mount_adapter))
        .route("/v1/adapters/:name", delete(routes::adapters::unmount_adapter))
        .route("/v1/adapters/:name/fuse", post(routes::finetune::fuse_adapter))
        // Fine-tuning
        .route("/v1/train", post(routes::train::train))
        .route("/v1/convert", post(routes::finetune::convert))
        // Anthropic-compatible
        .route("/v1/messages", post(routes::anthropic::messages))
        // Model cache + discovery
        .route("/api/models/local", get(routes::models::list_local_models))
        .route("/api/models/local/:org/*model", delete(routes::models::delete_local_model))
        .route("/api/huggingface/models", get(routes::models::search_hf_models))
        .route("/api/ps", get(routes::models::ps))
        // Prompt cache sessions
        .route("/v1/sessions/:session_id", delete(routes::chat::delete_session))
        // Benchmarking
        .route("/v1/benchmark", post(routes::benchmark::run_benchmark))
        // Reranking
        .route("/v1/rerank", post(routes::rerank::rerank))
        // Middleware
        .layer(middleware::from_fn(request_id_middleware))
        .layer(cors)
        .with_state(state.clone());

    // Warm prompts: load default model and pre-fill KV cache for known prompts
    if let Some(ref warm_file) = cfg.warm_prompts_file.clone() {
        let warm_state = state.clone();
        let warm_file = warm_file.clone();
        let default_model = cfg.default_model.clone();
        tokio::spawn(async move {
            if let Err(e) = warm_state.mlx.load_model(default_model, None).await {
                tracing::warn!("Warm-up: failed to load model: {}", e);
                return;
            }
            match tokio::fs::read_to_string(&warm_file).await {
                Ok(content) => match serde_json::from_str::<Vec<Vec<serde_json::Value>>>(&content) {
                    Ok(sets) => {
                        use crate::models::{ChatMessage, MessageContent};
                        let message_sets: Vec<Vec<ChatMessage>> = sets
                            .into_iter()
                            .map(|msgs| {
                                msgs.into_iter()
                                    .filter_map(|m| {
                                        let role = m["role"].as_str()?.to_string();
                                        let content = m["content"].as_str().unwrap_or("").to_string();
                                        Some(ChatMessage { role, content: MessageContent::Text(content) })
                                    })
                                    .collect()
                            })
                            .collect();
                        info!("Running warm-up for {} prompt set(s)", message_sets.len());
                        warm_state.mlx.warm_up(message_sets).await;
                    }
                    Err(e) => tracing::warn!("Warm-up: failed to parse {}: {}", warm_file, e),
                },
                Err(e) => tracing::warn!("Warm-up: cannot read {}: {}", warm_file, e),
            }
        });
    }

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse().expect("invalid address");
    info!("MLX LM Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
