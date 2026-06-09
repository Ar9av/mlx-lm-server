use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    Json,
};
use futures::StreamExt;
use crate::error::AudioError;
use crate::models::SpeechRequest;
use crate::state::AppState;

fn mime_for_format(fmt: &str) -> &'static str {
    match fmt {
        "mp3"  => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg"  => "audio/ogg",
        "opus" => "audio/ogg; codecs=opus",
        _      => "audio/wav",
    }
}

pub async fn create_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response, AudioError> {
    // Ensure TTS model loaded; auto-load default if not
    if state.audio.tts_model_id().await.is_none() {
        state.audio.load_tts(state.config.default_tts_model.clone()).await?;
    }

    let _permit = state.semaphore.acquire().await
        .map_err(|e| AudioError::Internal(e.to_string()))?;

    let voice = if req.voice.is_empty() { state.config.default_voice.clone() } else { req.voice.clone() };
    let speed = if req.speed <= 0.0 { state.config.default_speed } else { req.speed };
    let fmt = req.response_format.clone();
    let mime = mime_for_format(&fmt);

    if req.stream {
        let stream = state.audio.synthesize_stream(
            req.input,
            voice,
            speed,
            req.language,
        ).await?;

        let body = Body::from_stream(stream.map(|r| {
            r.map(|bytes| axum::body::Bytes::from(bytes))
             .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        }));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "audio/wav")
            .header("X-Accel-Buffering", "no")
            .header("Transfer-Encoding", "chunked")
            .body(body)
            .unwrap())
    } else {
        let bytes = state.audio.synthesize(
            req.input,
            voice,
            speed,
            req.language,
            req.ref_audio,
            req.ref_text,
            fmt,
        ).await?;

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime)
            .body(Body::from(bytes))
            .unwrap())
    }
}
