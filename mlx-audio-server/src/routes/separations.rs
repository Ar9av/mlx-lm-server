use axum::{
    body::Body,
    extract::{Multipart, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use std::io::Write;
use crate::error::AudioError;
use crate::models::SeparationResponse;
use crate::state::AppState;

/// POST /v1/audio/separations
/// Accepts multipart form: file (audio), description (text prompt for target sound)
/// Returns JSON with target + residual audio as base64-encoded WAV, or
/// streams the separated files as multipart when Accept: multipart/form-data
pub async fn create_separation(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<SeparationResponse>, AudioError> {
    if state.audio.sts_model_id().await.is_none() {
        return Err(AudioError::StsNotLoaded);
    }

    let mut audio_path: Option<String> = None;
    let mut description = "speech".to_string();

    while let Some(field) = multipart.next_field().await
        .map_err(|e| AudioError::Internal(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let bytes = field.bytes().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?;
                let mut tmp = tempfile::NamedTempFile::with_suffix(".wav")
                    .map_err(|e| AudioError::Internal(e.to_string()))?;
                tmp.write_all(&bytes)
                    .map_err(|e| AudioError::Internal(e.to_string()))?;
                let path = tmp.into_temp_path();
                audio_path = Some(path.to_string_lossy().to_string());
            }
            "description" => {
                description = field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?;
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let path = audio_path.ok_or_else(|| AudioError::Internal("No audio file provided".into()))?;

    let _permit = state.semaphore.acquire().await
        .map_err(|e| AudioError::Internal(e.to_string()))?;

    let target_tmp = tempfile::NamedTempFile::with_suffix(".wav")
        .map_err(|e| AudioError::Internal(e.to_string()))?;
    let residual_tmp = tempfile::NamedTempFile::with_suffix(".wav")
        .map_err(|e| AudioError::Internal(e.to_string()))?;

    let target_path = target_tmp.path().to_string_lossy().to_string();
    let residual_path = residual_tmp.path().to_string_lossy().to_string();

    state.audio.separate(
        path,
        description,
        target_path.clone(),
        residual_path.clone(),
    ).await?;

    // Read back and base64-encode
    let target_bytes = std::fs::read(&target_path)
        .map_err(|e| AudioError::AudioProcessing(e.to_string()))?;
    let residual_bytes = std::fs::read(&residual_path)
        .map_err(|e| AudioError::AudioProcessing(e.to_string()))?;

    let target_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &target_bytes);
    let residual_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &residual_bytes);

    Ok(Json(SeparationResponse {
        target_url: Some(format!("data:audio/wav;base64,{}", target_b64)),
        residual_url: Some(format!("data:audio/wav;base64,{}", residual_b64)),
        message: "Separation complete".into(),
    }))
}
