mod config;
mod error;
mod image_service;
mod models;
mod routes;
mod state;

use axum::{routing::{delete, get, post}, Router};
use state::AppState;
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

    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);
    let state = AppState::new(cfg.clone());

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/images/generations", post(routes::generations::create_image))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/models/load", post(routes::models::load_model))
        .route("/v1/models/unload", delete(routes::models::unload_model))
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    info!("mlx-image-server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
