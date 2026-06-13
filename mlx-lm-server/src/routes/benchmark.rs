use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::time::Instant;
use tracing::info;

use crate::mlx_service::MlxService;
use crate::models::{BenchmarkRequest, BenchmarkResult, ChatMessage, MessageContent, SamplerParams};
use crate::state::AppState;

pub async fn run_benchmark(
    State(state): State<AppState>,
    Json(req): Json<BenchmarkRequest>,
) -> impl IntoResponse {
    if !state.mlx.is_loaded().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "No model loaded. Load a model first via POST /v1/models/load"})),
        ).into_response();
    }

    let runs = req.runs.max(1).min(20);
    let max_tokens = req.max_tokens.max(1).min(512);
    let sampler = SamplerParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: 1.0,
        ..Default::default()
    };
    let model = state.mlx.current_model().await;

    info!("Running benchmark: {} runs, {} max_tokens", runs, max_tokens);

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: MessageContent::Text(req.prompt.clone()),
    }];

    let mut ttfts: Vec<f64> = Vec::new();
    let mut tps_list: Vec<f64> = Vec::new();
    let mut total_tokens = 0usize;

    let sem = state.inference_sem.clone();

    for i in 0..runs {
        let _permit = match sem.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": "inference queue closed"})),
                ).into_response()
            }
        };

        let run_start = Instant::now();
        let bench_id = MlxService::new_chat_id();
        match state.mlx.generate_stream(
            bench_id,
            messages.clone(),
            max_tokens,
            sampler.clone(),
            60.0,
            serde_json::Value::Object(Default::default()),
            None,
            None,
            None,
            None,
            vec![],
            false,
            0,
            None,
            None,
        ).await {
            Ok((mut stream, _)) => {
                use tokio_stream::StreamExt;
                let mut first_token = true;
                let mut token_count = 0usize;
                while let Some(result) = stream.next().await {
                    if result.is_ok() {
                        if first_token {
                            ttfts.push(run_start.elapsed().as_secs_f64() * 1000.0);
                            first_token = false;
                        }
                        token_count += 1;
                    }
                }
                let elapsed = run_start.elapsed().as_secs_f64();
                if token_count > 0 && elapsed > 0.0 {
                    tps_list.push(token_count as f64 / elapsed);
                }
                total_tokens += token_count;
                info!("Benchmark run {}/{}: {} tokens in {:.2}s", i + 1, runs, token_count, elapsed);
            }
            Err(e) => {
                return e.into_response();
            }
        }
    }

    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    tps_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50_idx = (ttfts.len() as f64 * 0.50) as usize;
    let p95_idx = (ttfts.len() as f64 * 0.95) as usize;
    let ttft_p50 = ttfts.get(p50_idx.min(ttfts.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let ttft_p95 = ttfts.get(p95_idx.min(ttfts.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let tps_mean = if tps_list.is_empty() { 0.0 } else { tps_list.iter().sum::<f64>() / tps_list.len() as f64 };
    let tps_p50 = tps_list.get(p50_idx.min(tps_list.len().saturating_sub(1))).copied().unwrap_or(0.0);

    (StatusCode::OK, Json(BenchmarkResult {
        model,
        runs,
        max_tokens,
        ttft_ms_p50: ttft_p50,
        ttft_ms_p95: ttft_p95,
        tokens_per_sec_mean: tps_mean,
        tokens_per_sec_p50: tps_p50,
        total_tokens_generated: total_tokens,
    })).into_response()
}
