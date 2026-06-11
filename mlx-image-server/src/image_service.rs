use crate::config::Config;
use crate::error::ImageError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

struct LoadedModel {
    flux: Py<PyAny>,
    model_id: String,
    quantize: Option<u32>,
    model_path: Option<String>,
    is_klein: bool,
}

fn is_klein_model(model_id: &str) -> bool {
    let lower = model_id.to_lowercase();
    lower.contains("klein") || lower.contains("flux.2") || lower.contains("flux2")
}

struct Inner {
    model: Option<LoadedModel>,
}

#[derive(Clone)]
pub struct ImageService {
    inner: Arc<Mutex<Inner>>,
    config: Arc<Config>,
}

impl ImageService {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { model: None })),
            config,
        }
    }

    pub async fn is_loaded(&self) -> bool {
        self.inner.lock().await.model.is_some()
    }

    pub async fn current_model(&self) -> Option<(String, Option<u32>, Option<String>)> {
        self.inner.lock().await.model.as_ref()
            .map(|m| (m.model_id.clone(), m.quantize, m.model_path.clone()))
    }

    pub async fn load(&self, model_id: String, quantize: Option<u32>, model_path: Option<String>) -> Result<(), ImageError> {
        info!("Loading image model: {} (quantize={:?}, model_path={:?})", model_id, quantize, model_path);
        let mid = model_id.clone();
        let mp = model_path.clone();
        let klein = is_klein_model(&mid);
        let flux_py = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Py<PyAny>> {
                let _ = py.import("mlx.core").and_then(|mx| mx.call_method0("synchronize"));

                let model_config_mod = py.import("mflux.models.common.config.model_config")?;

                let kwargs = PyDict::new(py);
                if let Some(ref path) = mp {
                    kwargs.set_item("model_path", path.as_str())?;
                }
                if let Some(q) = quantize {
                    kwargs.set_item("quantize", q as i64)?;
                }

                let flux = if klein {
                    let flux_mod = py.import("mflux.models.flux2.variants.txt2img.flux2_klein")?;
                    let model_config = model_config_mod.getattr("ModelConfig")?.call_method0("flux2_klein_4b")?;
                    kwargs.set_item("model_config", model_config)?;
                    flux_mod.getattr("Flux2Klein")?.call((), Some(kwargs))?
                } else {
                    let flux_mod = py.import("mflux.models.flux.variants.txt2img.flux")?;
                    let model_config = if mid.contains("dev") {
                        model_config_mod.getattr("ModelConfig")?.call_method0("dev")?
                    } else {
                        model_config_mod.getattr("ModelConfig")?.call_method0("schnell")?
                    };
                    kwargs.set_item("model_config", model_config)?;
                    flux_mod.getattr("Flux1")?.call((), Some(kwargs))?
                };

                Ok(flux.into())
            })
        })
        .await
        .map_err(|e| ImageError::Internal(e.to_string()))?
        .map_err(|e: PyErr| ImageError::LoadFailed(e.to_string()))?;

        let mut guard = self.inner.lock().await;
        guard.model = Some(LoadedModel { flux: flux_py, model_id, quantize, model_path, is_klein: klein });
        info!("Image model loaded");
        Ok(())
    }

    pub async fn unload(&self) {
        self.inner.lock().await.model = None;
    }

    pub async fn generate(
        &self,
        prompt: String,
        width: u32,
        height: u32,
        steps: u32,
        guidance: f64,
        seed: i64,
        negative_prompt: Option<String>,
    ) -> Result<Vec<u8>, ImageError> {
        let (flux, is_klein) = {
            let guard = self.inner.lock().await;
            match &guard.model {
                Some(m) => {
                    let f = Python::with_gil(|py| m.flux.clone_ref(py));
                    (f, m.is_klein)
                }
                None => return Err(ImageError::NotLoaded),
            }
        };

        let png_bytes = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<u8>> {
                let _ = py.import("mlx.core").and_then(|mx| mx.call_method0("synchronize"));

                let kwargs = PyDict::new(py);
                kwargs.set_item("seed", seed)?;
                kwargs.set_item("prompt", &prompt)?;
                kwargs.set_item("num_inference_steps", steps as i64)?;
                kwargs.set_item("width", width as i64)?;
                kwargs.set_item("height", height as i64)?;
                kwargs.set_item("guidance", guidance)?;
                if !is_klein {
                    if let Some(ref neg) = negative_prompt {
                        kwargs.set_item("negative_prompt", neg.as_str())?;
                    }
                }

                let result = flux.as_ref(py).call_method("generate_image", (), Some(kwargs))?;
                let pil_image = result.getattr("image")?;

                // Encode PIL image to PNG bytes via BytesIO
                let io = py.import("io")?;
                let buf = io.call_method0("BytesIO")?;
                pil_image.call_method("save", (buf, "PNG"), None)?;
                buf.call_method1("seek", (0i64,))?;
                let raw: Vec<u8> = buf.call_method0("read")?.extract()?;
                Ok(raw)
            })
        })
        .await
        .map_err(|e| ImageError::Internal(e.to_string()))?
        .map_err(|e: PyErr| ImageError::GenerationFailed(e.to_string()))?;

        Ok(png_bytes)
    }
}
