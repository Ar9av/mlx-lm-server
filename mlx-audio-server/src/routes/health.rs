use axum::{extract::State, Json};
use crate::models::HealthResponse;
use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        tts_model: state.audio.tts_model_id().await,
        stt_model: state.audio.stt_model_id().await,
        sts_model: state.audio.sts_model_id().await,
    })
}
