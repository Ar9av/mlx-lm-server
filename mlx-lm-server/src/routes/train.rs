use axum::{
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tracing::info;
use uuid::Uuid;

use crate::models::{TrainingJob, TrainProgress, TrainRequest};
use crate::state::{AppState, TrainingJobEntry};

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub async fn train(
    State(state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> Response {
    let job_id = format!("ftjob-{}", &Uuid::new_v4().to_string().replace('-', "")[..20]);
    let created_at = now_secs();

    info!(
        "Queuing training job {} — {} ({}) on {} for {} iters",
        job_id, req.model, req.fine_tune_type, req.data, req.iters.unwrap_or(100)
    );

    let (tx, _) = broadcast::channel::<String>(256);

    let job = TrainingJob {
        id: job_id.clone(),
        object: "fine_tuning.job",
        status: "pending".to_string(),
        model: req.model.clone(),
        fine_tune_type: req.fine_tune_type.clone(),
        adapter_path: req.adapter_path.clone(),
        created_at,
        started_at: None,
        completed_at: None,
        events: Vec::new(),
        error: None,
    };

    state.training_jobs.lock().await.insert(job_id.clone(), TrainingJobEntry {
        job: job.clone(),
        tx: tx.clone(),
    });

    let jobs = state.training_jobs.clone();
    let jid = job_id.clone();

    tokio::task::spawn_blocking(move || {
        let emit = |jobs: &tokio::sync::Mutex<std::collections::HashMap<String, TrainingJobEntry>>,
                    jid: &str,
                    tx: &broadcast::Sender<String>,
                    p: TrainProgress| {
            let line = serde_json::to_string(&p).unwrap_or_default();
            let _ = tx.send(line.clone());
            let rt = tokio::runtime::Handle::try_current();
            if let Ok(handle) = rt {
                handle.block_on(async {
                    let mut guard = jobs.lock().await;
                    if let Some(entry) = guard.get_mut(jid) {
                        entry.job.events.push(p);
                    }
                });
            }
        };

        // Mark running
        let rt = tokio::runtime::Handle::try_current();
        if let Ok(handle) = rt {
            handle.block_on(async {
                let mut guard = jobs.lock().await;
                if let Some(entry) = guard.get_mut(&jid) {
                    entry.job.status = "running".to_string();
                    entry.job.started_at = Some(now_secs());
                }
            });
        }

        Python::with_gil(|py| {
            if let Ok(mx) = py.import("mlx.core") {
                let _ = mx.call_method1("eval", (mx.call_method1("zeros", (1usize,)).ok(),));
            }

            let result: PyResult<()> = (|| {
                let args = PyDict::new(py);
                args.set_item("model", &req.model)?;
                args.set_item("train", true)?;
                args.set_item("data", &req.data)?;
                args.set_item("fine_tune_type", &req.fine_tune_type)?;
                args.set_item("adapter_path", &req.adapter_path)?;
                args.set_item("iters", req.iters.unwrap_or(100))?;
                args.set_item("batch_size", req.batch_size.unwrap_or(4))?;
                args.set_item("num_layers", req.num_layers.unwrap_or(16))?;
                args.set_item("max_seq_length", req.max_seq_length.unwrap_or(2048))?;
                args.set_item("learning_rate", req.learning_rate.unwrap_or(1e-4))?;
                args.set_item("steps_per_report", req.steps_per_report.unwrap_or(10))?;
                args.set_item("steps_per_eval", req.steps_per_eval.unwrap_or(200))?;
                args.set_item("save_every", req.save_every.unwrap_or(100))?;
                args.set_item("grad_accumulation_steps", req.grad_accumulation_steps.unwrap_or(1))?;
                args.set_item("mask_prompt", req.mask_prompt)?;
                args.set_item("grad_checkpoint", req.grad_checkpoint)?;
                args.set_item("test", req.test)?;
                args.set_item("optimizer", req.optimizer.as_deref().unwrap_or("adamw"))?;
                args.set_item("resume_adapter_file", req.resume_adapter_file.as_deref().map(|s| s.to_string()).unwrap_or_default())?;
                args.set_item("val_batches", 25i32)?;
                args.set_item("test_batches", 500i32)?;
                args.set_item("seed", 0i32)?;
                args.set_item("report_to", "none")?;
                args.set_item("project_name", "mlx-local-server")?;
                args.set_item("config", py.None())?;
                args.set_item("clear_cache_threshold", 0i32)?;

                let tuner = py.import("mlx_lm.tuner.trainer")?;
                let training_args_cls = tuner.getattr("TrainingArgs")?;
                let training_args = training_args_cls.call((), Some(args))?;

                emit(&jobs, &jid, &tx, TrainProgress {
                    event: "progress",
                    step: None,
                    loss: None,
                    val_loss: None,
                    tokens_per_sec: None,
                    adapter_path: None,
                    message: Some(format!(
                        "Training {} ({}) — {} iters, lr={:.0e}",
                        req.model, req.fine_tune_type,
                        req.iters.unwrap_or(100),
                        req.learning_rate.unwrap_or(1e-4)
                    )),
                });

                let lora_mod = py.import("mlx_lm.lora")?;
                lora_mod.call_method1("run", (training_args,))?;
                Ok(())
            })();

            let rt = tokio::runtime::Handle::try_current();
            match result {
                Ok(()) => {
                    emit(&jobs, &jid, &tx, TrainProgress {
                        event: "done",
                        step: None, loss: None, val_loss: None, tokens_per_sec: None,
                        adapter_path: Some(req.adapter_path.clone()),
                        message: Some("Training complete".into()),
                    });
                    if let Ok(handle) = rt {
                        handle.block_on(async {
                            let mut guard = jobs.lock().await;
                            if let Some(entry) = guard.get_mut(&jid) {
                                entry.job.status = "completed".to_string();
                                entry.job.completed_at = Some(now_secs());
                            }
                        });
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    emit(&jobs, &jid, &tx, TrainProgress {
                        event: "error",
                        step: None, loss: None, val_loss: None, tokens_per_sec: None,
                        adapter_path: None,
                        message: Some(msg.clone()),
                    });
                    if let Ok(handle) = rt {
                        handle.block_on(async {
                            let mut guard = jobs.lock().await;
                            if let Some(entry) = guard.get_mut(&jid) {
                                entry.job.status = "failed".to_string();
                                entry.job.error = Some(msg);
                                entry.job.completed_at = Some(now_secs());
                            }
                        });
                    }
                }
            }
        });
    });

    (StatusCode::ACCEPTED, Json(job)).into_response()
}

pub async fn get_training_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Response {
    let guard = state.training_jobs.lock().await;
    match guard.get(&job_id) {
        Some(entry) => (StatusCode::OK, Json(serde_json::to_value(&entry.job).unwrap_or_default())).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": {"message": "job not found", "code": "not_found"}}))).into_response(),
    }
}

pub async fn list_training_jobs(State(state): State<AppState>) -> impl IntoResponse {
    let guard = state.training_jobs.lock().await;
    let data: Vec<serde_json::Value> = guard.values()
        .map(|e| serde_json::to_value(&e.job).unwrap_or_default())
        .collect();
    Json(serde_json::json!({"object": "list", "data": data}))
}

pub async fn training_job_events(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Response {
    let (past_events, rx) = {
        let guard = state.training_jobs.lock().await;
        match guard.get(&job_id) {
            Some(entry) => {
                let events: Vec<String> = entry.job.events.iter()
                    .map(|e| serde_json::to_string(e).unwrap_or_default())
                    .collect();
                (events, entry.tx.subscribe())
            }
            None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "job not found"}))).into_response(),
        }
    };

    let past_stream = futures::stream::iter(
        past_events.into_iter().map(|line| {
            Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {}\n\n", line)))
        })
    );

    let live_stream = tokio_stream::StreamExt::map(
        BroadcastStream::new(rx),
        |item| match item {
            Ok(line) => Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {}\n\n", line))),
            Err(_) => Ok(Bytes::new()),
        }
    );

    let combined = futures::StreamExt::chain(past_stream, live_stream);

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));
    (StatusCode::OK, headers, axum::body::Body::from_stream(combined)).into_response()
}
