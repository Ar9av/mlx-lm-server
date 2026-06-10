use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    Json,
};
use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::models::{TrainProgress, TrainRequest};
use crate::state::AppState;

pub async fn train(
    State(_state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> impl IntoResponse {
    info!(
        "Training {} ({}) on {} for {} iters",
        req.model,
        req.fine_tune_type,
        req.data,
        req.iters.unwrap_or(100)
    );

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let adapter_path = req.adapter_path.clone();

    tokio::task::spawn_blocking(move || {
        let send = |p: TrainProgress| {
            let line = format!("data: {}\n\n", serde_json::to_string(&p).unwrap_or_default());
            let _ = tx.blocking_send(Ok(Bytes::from(line)));
        };

        Python::with_gil(|py| {
            // Ensure Metal stream is initialized on this thread
            if let Ok(mx) = py.import("mlx.core") {
                let _ = mx.call_method1("eval", (mx.call_method1("zeros", (1usize,)).ok(),));
            }

            let result: PyResult<()> = (|| {
                // Build args namespace for mlx_lm.lora
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
                if let Some(r) = &req.resume_adapter_file {
                    args.set_item("resume_adapter_file", r)?;
                } else {
                    args.set_item("resume_adapter_file", py.None())?;
                }
                // Fields required by TrainingArgs but not exposed per-request
                args.set_item("val_batches", 25i32)?;
                args.set_item("test_batches", 500i32)?;
                args.set_item("seed", 0i32)?;
                args.set_item("report_to", "none")?;
                args.set_item("project_name", "mlx-local-server")?;
                args.set_item("config", py.None())?;
                args.set_item("clear_cache_threshold", 0i32)?;

                // Build TrainingArgs from the dict
                let tuner = py.import("mlx_lm.tuner.trainer")?;
                let training_args_cls = tuner.getattr("TrainingArgs")?;
                let training_args = training_args_cls.call((), Some(args))?;

                send(TrainProgress {
                    event: "progress",
                    step: None,
                    loss: None,
                    val_loss: None,
                    tokens_per_sec: None,
                    adapter_path: None,
                    message: Some(format!(
                        "Training {} ({}) — {} iters, lr={:.0e}",
                        req.model,
                        req.fine_tune_type,
                        req.iters.unwrap_or(100),
                        req.learning_rate.unwrap_or(1e-4)
                    )),
                });

                let lora_mod = py.import("mlx_lm.lora")?;
                lora_mod.call_method1("run", (training_args,))?;

                Ok(())
            })();

            match result {
                Ok(()) => send(TrainProgress {
                    event: "done",
                    step: None,
                    loss: None,
                    val_loss: None,
                    tokens_per_sec: None,
                    adapter_path: Some(adapter_path),
                    message: Some("Training complete".into()),
                }),
                Err(e) => send(TrainProgress {
                    event: "error",
                    step: None,
                    loss: None,
                    val_loss: None,
                    tokens_per_sec: None,
                    adapter_path: None,
                    message: Some(e.to_string()),
                }),
            }
        });
    });

    let stream = ReceiverStream::new(rx);
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));
    (StatusCode::OK, headers, axum::body::Body::from_stream(stream)).into_response()
}
