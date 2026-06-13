mod audio_service;
mod config;
mod error;
mod models;
mod routes;
mod state;

use axum::{
    routing::{get, post, delete},
    Router,
};
use config::Config;
use state::AppState;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env();
    let addr = format!("{}:{}", config.host, config.port);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let state = AppState::new(config);

    let app = Router::new()
        // Health
        .route("/health", get(routes::health::health))
        // Model management
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/models/load", post(routes::models::load_model))
        .route("/v1/models/:type", delete(routes::models::unload_model))
        // TTS — OpenAI-compatible
        .route("/v1/audio/speech", post(routes::speech::create_speech))
        // STT — OpenAI-compatible
        .route("/v1/audio/transcriptions", post(routes::transcriptions::create_transcription))
        .route("/v1/audio/translations", post(routes::translations::create_translation))
        // STS — source separation
        .route("/v1/audio/separations", post(routes::separations::create_separation))
        // VAD — voice activity detection
        .route("/v1/audio/vad", post(routes::vad::detect_voice_activity))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    info!("mlx-audio-server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
