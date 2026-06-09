mod config;
mod error;
mod mlx_service;
mod models;
mod routes;
mod state;

use axum::{
    routing::{delete, get, post},
    Router,
};
use state::AppState;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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
        .route("/", get(routes::health::root))
        .route("/health", get(routes::health::health))
        .route("/status", get(routes::health::status))
        // OpenAI-compatible
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/models/load", post(routes::models::load_model))
        .route("/v1/models/*model_id", delete(routes::models::unload_model))
        // Local model cache
        .route("/api/models/local", get(routes::models::list_local_models))
        .route(
            "/api/models/local/:org/*model",
            delete(routes::models::delete_local_model),
        )
        // HuggingFace search
        .route(
            "/api/huggingface/models",
            get(routes::models::search_hf_models),
        )
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse().expect("invalid address");
    info!("MLX LM Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
