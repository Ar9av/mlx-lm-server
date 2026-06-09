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
        .route("/v1/models/*model_id", delete(routes::models::unload_model))
        // Adapter management
        .route("/v1/adapters", get(routes::adapters::list_adapters))
        .route("/v1/adapters/mount", post(routes::adapters::mount_adapter))
        .route("/v1/adapters/:name", delete(routes::adapters::unmount_adapter))
        // Anthropic-compatible
        .route("/v1/messages", post(routes::anthropic::messages))
        // Model cache + discovery
        .route("/api/models/local", get(routes::models::list_local_models))
        .route("/api/models/local/:org/*model", delete(routes::models::delete_local_model))
        .route("/api/huggingface/models", get(routes::models::search_hf_models))
        .route("/api/ps", get(routes::models::ps))
        // Middleware
        .layer(middleware::from_fn(request_id_middleware))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse().expect("invalid address");
    info!("MLX LM Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
