use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use base64::Engine;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::error::ImageError;
use crate::models::{ImageData, ImageGenerationRequest, ImageGenerationResponse};
use crate::state::AppState;

fn parse_size(size: &str) -> Result<(u32, u32), ImageError> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err(ImageError::InvalidSize(size.to_string()));
    }
    let w = parts[0].trim().parse::<u32>().map_err(|_| ImageError::InvalidSize(size.to_string()))?;
    let h = parts[1].trim().parse::<u32>().map_err(|_| ImageError::InvalidSize(size.to_string()))?;
    Ok((w, h))
}

pub async fn create_image(
    State(state): State<AppState>,
    Json(req): Json<ImageGenerationRequest>,
) -> impl IntoResponse {
    // Auto-load default model if nothing is loaded
    if !state.svc.is_loaded().await {
        let model_name = req.model.clone().unwrap_or_else(|| state.config.default_model.clone());
        // Default to 4-bit quantization to keep the large FLUX models in memory
        let quantize = req.quantize.or(state.config.default_quantize).or(Some(4));
        if let Err(e) = state.svc.load(model_name, quantize, None).await {
            return e.into_response();
        }
    }

    let (width, height) = match req.size.as_deref() {
        Some(s) => match parse_size(s) {
            Ok(wh) => wh,
            Err(e) => return e.into_response(),
        },
        None => (state.config.default_width, state.config.default_height),
    };

    let steps = req.steps.unwrap_or(state.config.default_steps);
    let guidance = req.guidance.unwrap_or(state.config.default_guidance);
    let n = req.n.unwrap_or(1).min(4);
    let want_b64 = req.response_format.as_deref() != Some("url");

    let _permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    info!("Generating {} image(s): {}x{}, steps={}, guidance={}", n, width, height, steps, guidance);

    let mut data = Vec::with_capacity(n as usize);
    for i in 0..n {
        let seed = req.seed.unwrap_or_else(|| {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().subsec_nanos() as i64
                + i as i64
        });

        match state.svc.generate(
            req.prompt.clone(),
            width,
            height,
            steps,
            guidance,
            seed,
            req.negative_prompt.clone(),
        ).await {
            Ok(png) => {
                if want_b64 {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
                    data.push(ImageData { b64_json: Some(encoded), url: None, revised_prompt: None });
                } else {
                    // url mode: return data URI (no file server needed)
                    let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
                    let uri = format!("data:image/png;base64,{}", encoded);
                    data.push(ImageData { b64_json: None, url: Some(uri), revised_prompt: None });
                }
            }
            Err(e) => return e.into_response(),
        }
    }

    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    (StatusCode::OK, Json(ImageGenerationResponse { created, data })).into_response()
}
