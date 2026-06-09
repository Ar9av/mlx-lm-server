use axum::{
    extract::{Multipart, State},
    Json,
};
use std::io::Write;
use crate::error::AudioError;
use crate::models::TranscriptionResponse;
use crate::state::AppState;

/// Transcribes audio and then (optionally) translates to English via mlx_audio STT
/// translate=true instructs the STT model to output English regardless of source language.
pub async fn create_translation(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<TranscriptionResponse>, AudioError> {
    if state.audio.stt_model_id().await.is_none() {
        state.audio.load_stt(state.config.default_stt_model.clone()).await?;
    }

    let mut tmp_file: Option<tempfile::NamedTempFile> = None;
    let mut audio_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut temperature = 0.0f64;

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
                audio_path = Some(tmp.path().to_string_lossy().to_string());
                tmp_file = Some(tmp);
            }
            "prompt" => {
                prompt = Some(field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?);
            }
            "temperature" => {
                temperature = field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?
                    .parse::<f64>()
                    .unwrap_or(0.0);
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let path = audio_path.ok_or_else(|| AudioError::Internal("No audio file provided".into()))?;

    let _permit = state.semaphore.acquire().await
        .map_err(|e| AudioError::Internal(e.to_string()))?;

    // Force language="en" so the model translates to English
    let (text, _, language) = state.audio.transcribe(
        path,
        Some("en".into()),
        prompt,
        temperature,
        false,
    ).await?;

    drop(tmp_file);
    Ok(Json(TranscriptionResponse { text, segments: None, language, duration: None }))
}
