use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::models::{
    BatchJob, BatchRequestCounts, BatchRequestItem, ChatCompletionRequest, CreateBatchRequest,
    SamplerParams, StopSequence, Tool,
};
use crate::state::AppState;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub async fn create_batch(
    State(state): State<AppState>,
    Json(req): Json<CreateBatchRequest>,
) -> Response {
    let batch_id = format!("batch_{}", Uuid::new_v4().to_string().replace('-', "")[..24].to_string());
    let created_at = now_secs();

    let job = BatchJob {
        id: batch_id.clone(),
        object: "batch",
        endpoint: req.endpoint.clone(),
        status: "validating".to_string(),
        created_at,
        in_progress_at: None,
        completed_at: None,
        failed_at: None,
        cancelled_at: None,
        request_counts: BatchRequestCounts {
            total: req.requests.len(),
            completed: 0,
            failed: 0,
        },
        requests: req.requests.iter()
            .map(|r| serde_json::to_value(r).unwrap_or(serde_json::Value::Null))
            .collect(),
        results: Vec::new(),
        errors: Vec::new(),
        metadata: req.metadata.clone(),
    };

    let batches = state.batches.clone();
    {
        let mut guard = batches.lock().await;
        guard.insert(batch_id.clone(), job);
    }

    let job_snapshot = {
        let guard = batches.lock().await;
        guard.get(&batch_id).cloned()
    };

    let requests: Vec<BatchRequestItem> = req.requests;
    let mlx = state.mlx.clone();
    let batches_bg = batches.clone();
    let bid = batch_id.clone();
    let default_max_tokens = state.config.default_max_tokens;
    let default_temperature = state.config.default_temperature;
    let default_top_p = state.config.default_top_p;
    let inference_sem = state.inference_sem.clone();

    tokio::spawn(async move {
        {
            let mut guard = batches_bg.lock().await;
            if let Some(job) = guard.get_mut(&bid) {
                job.status = "in_progress".to_string();
                job.in_progress_at = Some(now_secs());
            }
        }

        for item in requests {
            {
                let guard = batches_bg.lock().await;
                if let Some(job) = guard.get(&bid) {
                    if job.status == "cancelling" || job.status == "cancelled" {
                        break;
                    }
                }
            }

            let chat_req: ChatCompletionRequest = match serde_json::from_value(item.body.clone()) {
                Ok(r) => r,
                Err(e) => {
                    let err_entry = serde_json::json!({
                        "custom_id": item.custom_id,
                        "response": null,
                        "error": {
                            "message": format!("Failed to deserialize request: {}", e),
                            "code": "invalid_request"
                        }
                    });
                    let mut guard = batches_bg.lock().await;
                    if let Some(job) = guard.get_mut(&bid) {
                        job.errors.push(err_entry);
                        job.request_counts.failed += 1;
                    }
                    continue;
                }
            };

            let messages = chat_req.messages.clone();
            let max_tokens = chat_req.max_tokens.unwrap_or(default_max_tokens);
            let sampler = SamplerParams {
                temperature: chat_req.temperature.unwrap_or(default_temperature),
                top_p: chat_req.top_p.unwrap_or(default_top_p),
                top_k: chat_req.top_k,
                min_p: chat_req.min_p,
                repetition_penalty: chat_req.repetition_penalty,
                presence_penalty: chat_req.presence_penalty,
                frequency_penalty: chat_req.frequency_penalty,
                num_draft_tokens: chat_req.num_draft_tokens,
                xtc_probability: chat_req.xtc_probability,
                xtc_threshold: chat_req.xtc_threshold,
                thinking_budget: chat_req.thinking_budget,
                logit_bias: chat_req.logit_bias.clone(),
            };
            let chat_template_kwargs = chat_req.chat_template_kwargs.clone();
            let kv_bits = chat_req.kv_bits;
            let kv_group_size = chat_req.kv_group_size;
            let adapter_name = chat_req.adapter_name.clone();
            let tools: Option<Vec<Tool>> = chat_req.tools.clone();
            let stop_strings = match &chat_req.stop {
                Some(StopSequence::Single(s)) => vec![s.clone()],
                Some(StopSequence::Multiple(v)) => v.clone(),
                None => vec![],
            };
            let want_logprobs = chat_req.logprobs.unwrap_or(false);
            let top_n_logprobs = chat_req.top_logprobs.unwrap_or(0);
            let seed = chat_req.seed;
            let session_id = chat_req.session_id.clone();

            let _permit = match inference_sem.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    let err_entry = serde_json::json!({
                        "custom_id": item.custom_id,
                        "response": null,
                        "error": {
                            "message": "inference queue closed",
                            "code": "server_error"
                        }
                    });
                    let mut guard = batches_bg.lock().await;
                    if let Some(job) = guard.get_mut(&bid) {
                        job.errors.push(err_entry);
                        job.request_counts.failed += 1;
                    }
                    continue;
                }
            };

            match mlx.generate_response(
                messages,
                max_tokens,
                sampler,
                chat_template_kwargs,
                kv_bits,
                kv_group_size,
                adapter_name,
                tools,
                stop_strings,
                want_logprobs,
                top_n_logprobs,
                seed,
                session_id,
            ).await {
                Ok((content, prompt_tokens, completion_tokens, tool_calls, finish_reason, lp_list, reasoning)) => {
                    let created = now_secs();
                    let model_name = mlx.current_model().await.unwrap_or_else(|| chat_req.model.clone());

                    let message = if !tool_calls.is_empty() {
                        serde_json::json!({
                            "role": "assistant",
                            "content": null,
                            "tool_calls": tool_calls
                        })
                    } else {
                        let mut msg = serde_json::json!({
                            "role": "assistant",
                            "content": content
                        });
                        if let Some(r) = reasoning {
                            msg["reasoning_content"] = serde_json::Value::String(r);
                        }
                        msg
                    };

                    let mut choice = serde_json::json!({
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    });

                    if let Some(lps) = lp_list {
                        choice["logprobs"] = serde_json::json!({ "content": lps });
                    }

                    let response_body = serde_json::json!({
                        "id": format!("chatcmpl-{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [choice],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    });

                    let result_entry = serde_json::json!({
                        "custom_id": item.custom_id,
                        "response": {
                            "status_code": 200,
                            "body": response_body
                        },
                        "error": null
                    });

                    let mut guard = batches_bg.lock().await;
                    if let Some(job) = guard.get_mut(&bid) {
                        job.results.push(result_entry);
                        job.request_counts.completed += 1;
                    }
                }
                Err(e) => {
                    let err_entry = serde_json::json!({
                        "custom_id": item.custom_id,
                        "response": null,
                        "error": {
                            "message": e.to_string(),
                            "code": "server_error"
                        }
                    });
                    let mut guard = batches_bg.lock().await;
                    if let Some(job) = guard.get_mut(&bid) {
                        job.errors.push(err_entry);
                        job.request_counts.failed += 1;
                    }
                }
            }
        }

        let mut guard = batches_bg.lock().await;
        if let Some(job) = guard.get_mut(&bid) {
            if job.status == "cancelling" {
                job.status = "cancelled".to_string();
                job.cancelled_at = Some(now_secs());
            } else if job.request_counts.failed > 0 && job.request_counts.completed == 0 {
                job.status = "failed".to_string();
                job.failed_at = Some(now_secs());
            } else {
                job.status = "completed".to_string();
                job.completed_at = Some(now_secs());
            }
        }
    });

    match job_snapshot {
        Some(job) => (StatusCode::OK, Json(serde_json::to_value(&job).unwrap_or_default())).into_response(),
        None => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "failed to create batch"}))).into_response(),
    }
}

pub async fn get_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
) -> Response {
    let guard = state.batches.lock().await;
    match guard.get(&batch_id) {
        Some(job) => (StatusCode::OK, Json(serde_json::to_value(job).unwrap_or_default())).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("No batch found with id '{}'", batch_id),
                    "type": "invalid_request_error",
                    "code": "batch_not_found"
                }
            })),
        ).into_response(),
    }
}

pub async fn list_batches(State(state): State<AppState>) -> impl IntoResponse {
    let guard = state.batches.lock().await;
    let data: Vec<serde_json::Value> = guard.values()
        .map(|job| serde_json::to_value(job).unwrap_or_default())
        .collect();
    (StatusCode::OK, Json(serde_json::json!({ "object": "list", "data": data })))
}

pub async fn cancel_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
) -> Response {
    let mut guard = state.batches.lock().await;
    match guard.get_mut(&batch_id) {
        Some(job) => {
            if job.status == "in_progress" || job.status == "validating" {
                job.status = "cancelling".to_string();
            }
            (StatusCode::OK, Json(serde_json::to_value(&*job).unwrap_or_default())).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("No batch found with id '{}'", batch_id),
                    "type": "invalid_request_error",
                    "code": "batch_not_found"
                }
            })),
        ).into_response(),
    }
}
