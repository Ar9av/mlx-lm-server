use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use crate::error::AudioError;
use crate::models::{ModelList, ModelLoadRequest, ModelObject, ModelType};
use crate::state::AppState;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let mut data = Vec::new();
    if let Some(id) = state.audio.tts_model_id().await {
        data.push(ModelObject::new(id, ModelType::Tts));
    }
    if let Some(id) = state.audio.stt_model_id().await {
        data.push(ModelObject::new(id, ModelType::Stt));
    }
    if let Some(id) = state.audio.sts_model_id().await {
        data.push(ModelObject::new(id, ModelType::Sts));
    }
    Json(ModelList { object: "list", data })
}

pub async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<ModelLoadRequest>,
) -> Result<impl IntoResponse, AudioError> {
    match req.model_type {
        ModelType::Tts => state.audio.load_tts(req.model).await?,
        ModelType::Stt => state.audio.load_stt(req.model).await?,
        ModelType::Sts => state.audio.load_sts(req.model).await?,
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn unload_model(
    State(state): State<AppState>,
    axum::extract::Path(model_type): axum::extract::Path<String>,
) -> Result<impl IntoResponse, AudioError> {
    match model_type.as_str() {
        "tts" => state.audio.unload_tts().await,
        "stt" => state.audio.unload_stt().await,
        "sts" => state.audio.unload_sts().await,
        other => return Err(AudioError::Internal(format!("Unknown model type: {}", other))),
    }
    Ok(StatusCode::NO_CONTENT)
}
