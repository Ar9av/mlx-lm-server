use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use tracing::info;

use crate::models::{ConvertRequest, ConvertResponse, FuseRequest, FuseResponse};
use crate::state::AppState;

/// POST /v1/adapters/:name/fuse
/// Merge a mounted LoRA adapter into its base model weights.
pub async fn fuse_adapter(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<FuseRequest>,
) -> impl IntoResponse {
    // Resolve adapter path — accept a mounted adapter name or a raw path
    let adapter_path = {
        let adapters = state.mlx.list_adapters().await;
        if let Some(a) = adapters.iter().find(|a| a.name == name) {
            a.adapter_path.clone()
        } else {
            // treat `name` as a direct path
            name.clone()
        }
    };

    let base_model = match state.mlx.current_model().await {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "No model loaded. Load a model first via POST /v1/models/load"})),
            ).into_response()
        }
    };

    let output_path = req.output.clone().unwrap_or_else(|| format!("./fused-{name}"));
    let upload_repo = req.upload_repo.clone();
    let export_gguf = req.export_gguf;
    let dequantize = req.dequantize;
    let output_path_ret = output_path.clone();
    let upload_repo_ret = upload_repo.clone();

    info!("Fusing adapter '{}' ({}) into {} → {}", name, adapter_path, base_model, output_path);

    let result = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            if let Ok(mx) = py.import("mlx.core") {
                let _ = mx.call_method1("eval", (mx.call_method1("zeros", (1usize,)).ok(),));
            }

            let fuse = py.import("mlx_lm.fuse")?;

            // Build sys.argv equivalent — fuse.main() uses argparse
            let sys = py.import("sys")?;
            let mut argv = vec![
                "fuse".to_string(),
                "--model".to_string(),
                base_model.clone(),
                "--adapter-path".to_string(),
                adapter_path.clone(),
                "--save-path".to_string(),
                output_path.clone(),
            ];
            if dequantize {
                argv.push("--de-quantize".to_string());
            }
            if export_gguf {
                argv.push("--export-gguf".to_string());
            }
            if let Some(ref repo) = upload_repo {
                argv.push("--upload-repo".to_string());
                argv.push(repo.clone());
            }

            let py_argv = PyList::new(py, &argv);
            sys.setattr("argv", py_argv)?;
            fuse.call_method0("main")?;

            Ok::<(), PyErr>(())
        })
    })
    .await;

    match result {
        Ok(Ok(())) => {
            let gguf_path = if export_gguf {
                Some(format!("{output_path_ret}/model.gguf"))
            } else {
                None
            };
            (
                StatusCode::OK,
                Json(FuseResponse {
                    output_path: output_path_ret,
                    gguf_path,
                    uploaded_to: upload_repo_ret,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// POST /v1/convert
/// Convert a HuggingFace or GGUF model to MLX format, with optional quantization.
pub async fn convert(
    State(_state): State<AppState>,
    Json(req): Json<ConvertRequest>,
) -> impl IntoResponse {
    let output_path = req.output.clone().unwrap_or_else(|| {
        let slug = req.model.split('/').last().unwrap_or("model");
        format!("./mlx-{slug}")
    });
    let upload_repo_ret = req.upload_repo.clone();
    let output_path_ret = output_path.clone();

    info!("Converting {} → {}", req.model, output_path);

    let result = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            if let Ok(mx) = py.import("mlx.core") {
                let _ = mx.call_method1("eval", (mx.call_method1("zeros", (1usize,)).ok(),));
            }

            let sys = py.import("sys")?;
            let mut argv = vec![
                "convert".to_string(),
                "--hf-path".to_string(),
                req.model.clone(),
                "--mlx-path".to_string(),
                output_path.clone(),
            ];
            if let Some(bits) = req.quantize_bits {
                argv.push("-q".to_string());
                argv.push("--q-bits".to_string());
                argv.push(bits.to_string());
            }
            if let Some(gs) = req.quantize_group_size {
                argv.push("--q-group-size".to_string());
                argv.push(gs.to_string());
            }
            if let Some(ref repo) = req.upload_repo {
                argv.push("--upload-repo".to_string());
                argv.push(repo.clone());
            }
            if let Some(ref tok) = req.hf_token {
                argv.push("--hf-token".to_string());
                argv.push(tok.clone());
            }

            let py_argv = PyList::new(py, &argv);
            sys.setattr("argv", py_argv)?;

            let convert_mod = py.import("mlx_lm.convert")?;
            convert_mod.call_method0("main")?;

            Ok::<(), PyErr>(())
        })
    })
    .await;

    match result {
        Ok(Ok(())) => (
            StatusCode::OK,
            Json(ConvertResponse {
                output_path: output_path_ret,
                uploaded_to: upload_repo_ret,
            }),
        )
            .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
