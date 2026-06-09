use crate::config::Config;
use crate::error::MlxError;
use crate::models::ChatMessage;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn, error};
use uuid::Uuid;

pub const MAX_TOKEN_LIMIT: usize = 4096;
pub const MAX_WINDOW_TOKENS: usize = 3000;
pub const MAX_MESSAGE_TOKENS: usize = 4096;

struct LoadedModel {
    model: Py<PyAny>,
    tokenizer: Py<PyAny>,
    model_id: String,
}

struct Inner {
    loaded: Option<LoadedModel>,
}

#[derive(Clone)]
pub struct MlxService {
    inner: Arc<Mutex<Inner>>,
    config: Arc<Config>,
}

impl MlxService {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { loaded: None })),
            config,
        }
    }

    pub async fn current_model(&self) -> Option<String> {
        self.inner.lock().await.loaded.as_ref().map(|l| l.model_id.clone())
    }

    pub async fn is_loaded(&self) -> bool {
        self.inner.lock().await.loaded.is_some()
    }

    pub async fn load_model(&self, model_id: String) -> Result<(), MlxError> {
        {
            let guard = self.inner.lock().await;
            if guard.loaded.as_ref().map(|l| l.model_id == model_id).unwrap_or(false) {
                return Ok(());
            }
        }

        if model_id.trim().is_empty() {
            return Err(MlxError::LoadFailed("model name cannot be empty".into()));
        }

        if let Some(allowed) = &self.config.allowed_models {
            if !allowed.contains(&model_id) {
                return Err(MlxError::NotAllowed(model_id));
            }
        }

        if let Some(max_gb) = self.config.max_model_size_gb {
            check_model_size(&model_id, max_gb).await?;
        }

        info!("Loading model: {}", model_id);
        let mid = model_id.clone();
        let (model_py, tokenizer_py) = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(PyObject, PyObject)> {
                let mlx_lm = py.import("mlx_lm")?;
                let result = mlx_lm.getattr("load")?.call1((&mid,))?;
                let model: PyObject = result.get_item(0)?.into();
                let tokenizer: PyObject = result.get_item(1)?.into();
                Ok((model, tokenizer))
            })
        })
        .await
        .map_err(|e| MlxError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| MlxError::LoadFailed(e.to_string()))?;

        let mut guard = self.inner.lock().await;
        guard.loaded = Some(LoadedModel {
            model: model_py,
            tokenizer: tokenizer_py,
            model_id,
        });
        info!("Model loaded successfully");
        Ok(())
    }

    pub async fn unload_model(&self) -> Option<String> {
        let mut guard = self.inner.lock().await;
        if let Some(loaded) = guard.loaded.take() {
            let id = loaded.model_id.clone();
            drop(loaded);
            tokio::task::spawn_blocking(|| {
                let _ = Python::with_gil(|py| -> PyResult<()> {
                    py.import("mlx.core")?.call_method0("clear_cache")?;
                    Ok(())
                });
            });
            info!("Model unloaded: {}", id);
            Some(id)
        } else {
            None
        }
    }

    pub async fn generate_response(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        chat_template_kwargs: serde_json::Value,
    ) -> Result<(String, usize, usize), MlxError> {
        let (model_py, tokenizer_py) = self.get_model_refs().await?;
        let max_tokens = max_tokens.min(MAX_TOKEN_LIMIT).max(1);

        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(String, usize, usize)> {
                let tokenizer = tokenizer_py.as_ref(py);
                let windowed = apply_sliding_window(messages, MAX_WINDOW_TOKENS, py, tokenizer);
                let prompt = apply_chat_template(py, tokenizer, &windowed, &chat_template_kwargs)?;
                let prompt_tokens = count_tokens(py, tokenizer, &prompt);

                let mlx_lm = py.import("mlx_lm")?;
                let sampler = make_sampler(py, temperature, top_p)?;

                let kwargs = PyDict::new(py);
                kwargs.set_item("prompt", &prompt)?;
                kwargs.set_item("max_tokens", max_tokens as i64)?;
                kwargs.set_item("sampler", sampler)?;

                let response: String = mlx_lm
                    .getattr("generate")?
                    .call((model_py.as_ref(py), tokenizer), Some(kwargs))?
                    .extract()?;

                let completion_tokens = count_tokens(py, tokenizer, &response);
                Ok((response, prompt_tokens, completion_tokens))
            })
        })
        .await
        .map_err(|e| MlxError::Internal(e.to_string()))?
        .map_err(|e: PyErr| MlxError::Python(e.to_string()))?;

        Ok(result)
    }

    pub async fn generate_stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        timeout_secs: f64,
        chat_template_kwargs: serde_json::Value,
    ) -> Result<ReceiverStream<Result<String, MlxError>>, MlxError> {
        let (model_py, tokenizer_py) = self.get_model_refs().await?;
        let max_tokens = max_tokens.min(MAX_TOKEN_LIMIT).max(1);
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, MlxError>>(64);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let run = || -> PyResult<()> {
                    let tokenizer = tokenizer_py.as_ref(py);
                    let windowed = apply_sliding_window(messages, MAX_WINDOW_TOKENS, py, tokenizer);
                    let prompt =
                        apply_chat_template(py, tokenizer, &windowed, &chat_template_kwargs)?;

                    let mlx_lm = py.import("mlx_lm")?;
                    let sampler = make_sampler(py, temperature, top_p)?;

                    let kwargs = PyDict::new(py);
                    kwargs.set_item("prompt", &prompt)?;
                    kwargs.set_item("max_tokens", max_tokens as i64)?;
                    kwargs.set_item("sampler", sampler)?;

                    let generator = mlx_lm
                        .getattr("stream_generate")?
                        .call((model_py.as_ref(py), tokenizer), Some(kwargs))?;

                    let start = Instant::now();
                    for item in generator.iter()? {
                        if start.elapsed().as_secs_f64() > timeout_secs {
                            warn!("Stream timeout exceeded");
                            let _ = tx.blocking_send(Err(MlxError::Timeout));
                            return Ok(());
                        }
                        let text: String = item?.getattr("text")?.extract()?;
                        if tx.blocking_send(Ok(text)).is_err() {
                            break;
                        }
                    }
                    Ok(())
                };

                if let Err(e) = run() {
                    error!("Streaming Python error: {}", e);
                    let _ = tx.blocking_send(Err(MlxError::Python(e.to_string())));
                }
            });
        });

        Ok(ReceiverStream::new(rx))
    }

    async fn get_model_refs(&self) -> Result<(Py<PyAny>, Py<PyAny>), MlxError> {
        let guard = self.inner.lock().await;
        let loaded = guard.loaded.as_ref().ok_or(MlxError::NotLoaded)?;
        let (m, t) = Python::with_gil(|py| {
            (loaded.model.clone_ref(py), loaded.tokenizer.clone_ref(py))
        });
        Ok((m, t))
    }

    pub fn new_chat_id() -> String {
        format!("chatcmpl-{}", &Uuid::new_v4().to_string().replace('-', "")[..12])
    }
}

// ── Pure helpers (called inside spawn_blocking / with_gil) ───────────────────

fn prepare_messages(messages: &[ChatMessage]) -> Vec<HashMap<String, String>> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut rest: Vec<HashMap<String, String>> = Vec::new();

    for msg in messages {
        if msg.role == "system" {
            system_parts.push(msg.content.clone());
        } else {
            let mut m = HashMap::new();
            m.insert("role".into(), msg.role.clone());
            m.insert("content".into(), msg.content.clone());
            rest.push(m);
        }
    }

    if !system_parts.is_empty() {
        if let Some(first) = rest.first_mut() {
            if first.get("role").map(|r| r == "user").unwrap_or(false) {
                let prefix = system_parts.join("\n");
                let content = first.get("content").cloned().unwrap_or_default();
                first.insert("content".into(), format!("{}\n\n{}", prefix, content));
            }
        }
    }
    rest
}

fn apply_sliding_window(
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    py: Python,
    tokenizer: &PyAny,
) -> Vec<ChatMessage> {
    let system: Vec<ChatMessage> = messages.iter().filter(|m| m.role == "system").cloned().collect();
    let non_system: Vec<ChatMessage> = messages.into_iter().filter(|m| m.role != "system").collect();

    let total: usize = non_system
        .iter()
        .map(|m| count_tokens(py, tokenizer, &m.content))
        .sum();

    if total <= max_tokens {
        let mut result = system;
        result.extend(non_system);
        return result;
    }

    let mut kept: Vec<ChatMessage> = Vec::new();
    let mut used = 0usize;
    for msg in non_system.iter().rev() {
        let t = count_tokens(py, tokenizer, &msg.content);
        if used + t > max_tokens {
            break;
        }
        used += t;
        kept.insert(0, msg.clone());
    }

    if kept.is_empty() {
        if let Some(last) = non_system.last() {
            kept.push(last.clone());
        }
    }

    debug!("Sliding window: kept {} messages ({} tokens)", kept.len(), used);
    let mut result = system;
    result.extend(kept);
    result
}

fn apply_chat_template(
    py: Python,
    tokenizer: &PyAny,
    messages: &[ChatMessage],
    extra_kwargs: &serde_json::Value,
) -> PyResult<String> {
    let prepared = prepare_messages(messages);

    let py_messages = PyList::new(
        py,
        prepared.iter().map(|m| {
            let d = PyDict::new(py);
            d.set_item("role", m.get("role").unwrap_or(&String::new())).unwrap();
            d.set_item("content", m.get("content").unwrap_or(&String::new())).unwrap();
            d
        }),
    );

    let has_template = tokenizer
        .getattr("chat_template")
        .map(|v| !v.is_none())
        .unwrap_or(false);

    if has_template && tokenizer.hasattr("apply_chat_template")? {
        let kwargs = PyDict::new(py);
        kwargs.set_item("tokenize", false)?;
        kwargs.set_item("add_generation_prompt", true)?;

        if let Some(obj) = extra_kwargs.as_object() {
            for (k, v) in obj {
                let py_val = json_to_py(py, v)?;
                kwargs.set_item(k.as_str(), py_val)?;
            }
        }

        let result: String = tokenizer
            .call_method("apply_chat_template", (py_messages,), Some(kwargs))?
            .extract()?;
        Ok(result)
    } else {
        let lines: Vec<String> = prepared
            .iter()
            .map(|m| {
                let role = if m.get("role").map(|r| r == "user").unwrap_or(false) {
                    "User"
                } else {
                    "Assistant"
                };
                format!("{}: {}", role, m.get("content").unwrap_or(&String::new()))
            })
            .collect();
        Ok(format!("{}\nAssistant: ", lines.join("\n")))
    }
}

fn make_sampler<'py>(py: Python<'py>, temperature: f64, top_p: f64) -> PyResult<&'py PyAny> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("temp", temperature)?;
    kwargs.set_item("top_p", top_p)?;
    py.import("mlx_lm.sample_utils")?
        .getattr("make_sampler")?
        .call((), Some(kwargs))
}

fn count_tokens(py: Python, tokenizer: &PyAny, text: &str) -> usize {
    tokenizer
        .call_method1("encode", (text,))
        .and_then(|t| t.len())
        .unwrap_or(text.len() / 4)
}

fn json_to_py<'py>(py: Python<'py>, val: &serde_json::Value) -> PyResult<&'py PyAny> {
    Ok(match val {
        serde_json::Value::Null => py.None().into_ref(py),
        serde_json::Value::Bool(b) => b.into_py(py).into_ref(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py).into_ref(py)
            } else {
                n.as_f64().unwrap_or(0.0).into_py(py).into_ref(py)
            }
        }
        serde_json::Value::String(s) => s.into_py(py).into_ref(py),
        serde_json::Value::Array(arr) => {
            let list = PyList::new(py, arr.iter().map(|v| json_to_py(py, v).unwrap()));
            list.as_ref()
        }
        serde_json::Value::Object(obj) => {
            let d = PyDict::new(py);
            for (k, v) in obj {
                d.set_item(k.as_str(), json_to_py(py, v)?)?;
            }
            d.as_ref()
        }
    })
}

async fn check_model_size(model_id: &str, max_gb: f64) -> Result<(), MlxError> {
    let url = format!("https://huggingface.co/api/models/{}", model_id);
    let client = reqwest::Client::new();
    let resp: serde_json::Value = match client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(r) => match r.json().await {
            Ok(v) => v,
            Err(_) => return Ok(()),
        },
        Err(_) => return Ok(()),
    };

    if let Some(bytes) = resp.get("usedStorage").and_then(|v| v.as_f64()) {
        let size_gb = bytes / (1024.0_f64.powi(3));
        if size_gb > max_gb {
            return Err(MlxError::TooLarge(model_id.to_string(), size_gb, max_gb));
        }
    }
    Ok(())
}
