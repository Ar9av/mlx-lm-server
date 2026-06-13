use axum::{
    extract::{Multipart, State},
    Json,
};
use crate::error::AudioError;
use crate::models::{VadResponse, VadSegment};
use crate::state::AppState;
use std::io::Write;

pub async fn detect_voice_activity(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<VadResponse>, AudioError> {
    // Auto-load default VAD model if none loaded
    if state.audio.vad_model_id().await.is_none() {
        state.audio.load_vad(state.config.default_vad_model.clone()).await?;
    }

    let mut tmp_file: Option<tempfile::NamedTempFile> = None;
    let mut audio_path: Option<String> = None;
    let mut threshold: Option<f64> = None;
    let mut min_speech_ms: Option<u64> = None;
    let mut min_silence_ms: Option<u64> = None;
    let mut speech_pad_ms: Option<u64> = None;
    let mut return_seconds = true;

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
            "threshold" => {
                threshold = field.text().await.ok().and_then(|s| s.parse().ok());
            }
            "min_speech_duration_ms" => {
                min_speech_ms = field.text().await.ok().and_then(|s| s.parse().ok());
            }
            "min_silence_duration_ms" => {
                min_silence_ms = field.text().await.ok().and_then(|s| s.parse().ok());
            }
            "speech_pad_ms" => {
                speech_pad_ms = field.text().await.ok().and_then(|s| s.parse().ok());
            }
            "return_seconds" => {
                return_seconds = field.text().await
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let path = audio_path.ok_or_else(|| AudioError::Internal("No audio file provided".into()))?;
    let model_name = state.audio.vad_model_id().await.unwrap_or_default();

    let _permit = state.semaphore.acquire().await
        .map_err(|e| AudioError::Internal(e.to_string()))?;

    let (raw_segments, duration) = state.audio.detect_speech_segments(
        path,
        threshold,
        min_speech_ms,
        min_silence_ms,
        speech_pad_ms,
        return_seconds,
    ).await?;

    drop(tmp_file);

    let segments = raw_segments.into_iter()
        .map(|(start, end)| VadSegment { start, end })
        .collect();

    Ok(Json(VadResponse { model: model_name, segments, duration }))
}
