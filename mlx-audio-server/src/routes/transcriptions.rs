use axum::{
    extract::{Multipart, State},
    Json,
};
use crate::error::AudioError;
use crate::models::{TranscriptionParams, TranscriptionResponse};
use crate::state::AppState;
use std::io::Write;

pub async fn create_transcription(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<TranscriptionResponse>, AudioError> {
    // Ensure STT model loaded
    if state.audio.stt_model_id().await.is_none() {
        state.audio.load_stt(state.config.default_stt_model.clone()).await?;
    }

    let mut audio_path: Option<String> = None;
    let mut params = TranscriptionParams {
        model: None,
        language: None,
        response_format: "json".into(),
        prompt: None,
        temperature: None,
        timestamp_granularities: None,
    };

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
                // Don't drop path here — keep alive until after transcription
                // (We transfer ownership into audio_path as String, so file stays on disk until OS cleans tmp)
            }
            "model" => {
                params.model = Some(field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?);
            }
            "language" => {
                params.language = Some(field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?);
            }
            "prompt" => {
                params.prompt = Some(field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?);
            }
            "temperature" => {
                let t = field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?;
                params.temperature = t.parse::<f64>().ok();
            }
            "response_format" => {
                params.response_format = field.text().await
                    .map_err(|e| AudioError::Internal(e.to_string()))?;
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let path = audio_path.ok_or_else(|| AudioError::Internal("No audio file provided".into()))?;

    let with_segments = params.timestamp_granularities
        .as_ref()
        .map(|g| g.iter().any(|s| s == "segment"))
        .unwrap_or(false);

    let _permit = state.semaphore.acquire().await
        .map_err(|e| AudioError::Internal(e.to_string()))?;

    let (text, segments, language) = state.audio.transcribe(
        path,
        params.language,
        params.prompt,
        params.temperature.unwrap_or(0.0),
        with_segments,
    ).await?;

    let segs = if with_segments && !segments.is_empty() { Some(segments) } else { None };

    Ok(Json(TranscriptionResponse { text, segments: segs, language, duration: None }))
}
