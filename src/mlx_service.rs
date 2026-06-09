use crate::config::Config;
use crate::error::MlxError;
use crate::models::{ChatMessage, MountedAdapterInfo};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn, error};
use uuid::Uuid;

pub const MAX_TOKEN_LIMIT: usize = 32768;
pub const MAX_WINDOW_TOKENS: usize = 3000;
pub const MAX_MESSAGE_TOKENS: usize = 4096;

struct LoadedModel {
    model: Py<PyAny>,
    tokenizer: Py<PyAny>,
    model_id: String,
    loaded_at: Instant,
    loaded_at_unix: u64,
    draft_model: Option<Py<PyAny>>,
}

const MAX_MOUNTED_ADAPTERS: usize = 4;

struct MountedAdapter {
    model: Py<PyAny>,
    tokenizer: Py<PyAny>,
    base_model_id: String,
    adapter_path: String,
    mounted_at: u64,
}

struct Inner {
    loaded: Option<LoadedModel>,
    adapters: HashMap<String, MountedAdapter>,
}

#[derive(Clone)]
pub struct MlxService {
    inner: Arc<Mutex<Inner>>,
    config: Arc<Config>,
}

impl MlxService {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { loaded: None, adapters: HashMap::new() })),
            config,
        }
    }

    pub async fn current_model(&self) -> Option<String> {
        self.inner.lock().await.loaded.as_ref().map(|l| l.model_id.clone())
    }

    pub async fn is_loaded(&self) -> bool {
        self.inner.lock().await.loaded.is_some()
    }

    pub async fn ps_info(&self) -> Option<(String, u64, f64)> {
        let guard = self.inner.lock().await;
        guard.loaded.as_ref().map(|l| {
            let rss = process_rss_mb();
            (l.model_id.clone(), l.loaded_at_unix, rss)
        })
    }

    pub async fn load_model(&self, model_id: String, adapter: Option<String>) -> Result<(), MlxError> {
        self.load_model_with_drafter(model_id, adapter, None).await
    }

    pub async fn load_model_with_drafter(
        &self,
        model_id: String,
        adapter: Option<String>,
        drafter: Option<String>,
    ) -> Result<(), MlxError> {
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

        let model_size_gb = fetch_model_size_gb(&model_id).await;

        if let Some(size_gb) = model_size_gb {
            if let Some(max_gb) = self.config.max_model_size_gb {
                if size_gb > max_gb {
                    return Err(MlxError::TooLarge(model_id.clone(), size_gb, max_gb));
                }
            }

            if let Some(avail_gb) = available_ram_gb() {
                if size_gb > avail_gb * 0.9 {
                    warn!(
                        "Model '{}' needs ~{:.1}GB but only {:.1}GB available — may OOM",
                        model_id, size_gb, avail_gb
                    );
                    if size_gb > avail_gb * 1.5 {
                        return Err(MlxError::InsufficientRam(model_id.clone(), size_gb, avail_gb));
                    }
                }
            }
        }

        info!("Loading model: {} (adapter: {:?}, drafter: {:?})", model_id, adapter, drafter);
        let mid = model_id.clone();
        let (model_py, tokenizer_py, draft_model_py) = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(PyObject, PyObject, Option<PyObject>)> {
                let mlx_lm = py.import("mlx_lm")?;
                let kwargs = PyDict::new(py);
                if let Some(path) = &adapter {
                    kwargs.set_item("adapter_path", path.as_str())?;
                }
                let result = mlx_lm.getattr("load")?.call((&mid,), Some(kwargs))?;
                let model: PyObject = result.get_item(0)?.into();
                let tokenizer: PyObject = result.get_item(1)?.into();

                let draft = if let Some(ref drafter_id) = drafter {
                    info!("Loading draft model: {}", drafter_id);
                    let d_result = mlx_lm.getattr("load")?.call((drafter_id.as_str(),), None)?;
                    let d_model: PyObject = d_result.get_item(0)?.into();
                    Some(d_model)
                } else {
                    None
                };

                Ok((model, tokenizer, draft))
            })
        })
        .await
        .map_err(|e| MlxError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| MlxError::LoadFailed(e.to_string()))?;

        let loaded_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut guard = self.inner.lock().await;
        guard.loaded = Some(LoadedModel {
            model: model_py,
            tokenizer: tokenizer_py,
            model_id,
            loaded_at: Instant::now(),
            loaded_at_unix,
            draft_model: draft_model_py,
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

    pub async fn tokenize(&self, text: String) -> Result<Vec<i64>, MlxError> {
        let (_, tokenizer_py, _) = self.get_model_refs().await?;
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<i64>> {
                tokenizer_py.as_ref(py).call_method1("encode", (text.as_str(),))?.extract()
            })
        })
        .await
        .map_err(|e| MlxError::Internal(e.to_string()))?
        .map_err(|e: PyErr| MlxError::Python(e.to_string()))
    }

    pub async fn get_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MlxError> {
        let (model_py, tokenizer_py, _) = self.get_model_refs().await?;
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<Vec<f32>>> {
                let model = model_py.as_ref(py);
                let tokenizer = tokenizer_py.as_ref(py);
                let mx = py.import("mlx.core")?;

                // Find the embedding layer (handles LLaMA/Mistral/GPT-2 architectures)
                let embed_layer = ["embed_tokens", "embedding", "wte"]
                    .iter()
                    .find_map(|attr: &&str| {
                        model.getattr("model").ok()
                            .and_then(|m| m.getattr(*attr).ok())
                            .or_else(|| model.getattr(*attr).ok())
                    })
                    .ok_or_else(|| pyo3::exceptions::PyAttributeError::new_err(
                        "Cannot find embedding layer; model must have embed_tokens, embedding, or wte",
                    ))?;

                let mut results = Vec::new();
                for text in &texts {
                    let tokens: Vec<i64> = tokenizer.call_method1("encode", (text.as_str(),))?.extract()?;
                    let input_ids = mx.call_method1("array", (vec![tokens],))?;
                    let embeds = embed_layer.call1((input_ids,))?; // [1, seq, dim]

                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", 1)?;
                    let pooled = mx.call_method("mean", (embeds,), Some(kwargs))?; // [1, dim]
                    mx.call_method1("eval", (pooled,))?;

                    let rows: Vec<Vec<f32>> = pooled.call_method0("tolist")?.extract()?;
                    let mut emb = rows.into_iter().next().unwrap_or_default();

                    // L2 normalize
                    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        emb.iter_mut().for_each(|x| *x /= norm);
                    }
                    results.push(emb);
                }
                Ok(results)
            })
        })
        .await
        .map_err(|e| MlxError::Internal(e.to_string()))?
        .map_err(|e: PyErr| MlxError::Python(e.to_string()))
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        kv_bits: Option<u32>,
        kv_group_size: Option<u32>,
        adapter_name: Option<String>,
    ) -> Result<(String, usize, usize), MlxError> {
        let (model_py, tokenizer_py, _) = self.get_model_refs_for(adapter_name.as_deref()).await?;
        let max_tokens = max_tokens.min(MAX_TOKEN_LIMIT).max(1);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(String, usize, usize)> {
                let tokenizer = tokenizer_py.as_ref(py);
                let prompt_tokens = count_tokens(py, tokenizer, &prompt);
                let mlx_lm = py.import("mlx_lm")?;
                let sampler = make_sampler(py, temperature, top_p)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("prompt", prompt.as_str())?;
                kwargs.set_item("max_tokens", max_tokens as i64)?;
                kwargs.set_item("sampler", sampler)?;
                if let Some(bits) = kv_bits {
                    kwargs.set_item("kv_bits", bits)?;
                    kwargs.set_item("kv_group_size", kv_group_size.unwrap_or(64))?;
                }
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
        .map_err(|e: PyErr| MlxError::Python(e.to_string()))
    }

    pub async fn generate_response(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        chat_template_kwargs: serde_json::Value,
        kv_bits: Option<u32>,
        kv_group_size: Option<u32>,
        adapter_name: Option<String>,
    ) -> Result<(String, usize, usize), MlxError> {
        let (model_py, tokenizer_py, draft_model_py) = self.get_model_refs_for(adapter_name.as_deref()).await?;
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
                if let Some(bits) = kv_bits {
                    kwargs.set_item("kv_bits", bits)?;
                    kwargs.set_item("kv_group_size", kv_group_size.unwrap_or(64))?;
                }
                if let Some(ref draft) = draft_model_py {
                    kwargs.set_item("draft_model", draft.as_ref(py))?;
                }

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
        kv_bits: Option<u32>,
        kv_group_size: Option<u32>,
        adapter_name: Option<String>,
    ) -> Result<ReceiverStream<Result<String, MlxError>>, MlxError> {
        let (model_py, tokenizer_py, draft_model_py) = self.get_model_refs_for(adapter_name.as_deref()).await?;
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
                    if let Some(bits) = kv_bits {
                        kwargs.set_item("kv_bits", bits)?;
                        kwargs.set_item("kv_group_size", kv_group_size.unwrap_or(64))?;
                    }
                    if let Some(ref draft) = draft_model_py {
                        kwargs.set_item("draft_model", draft.as_ref(py))?;
                    }

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

    pub async fn generate_vision_response(
        &self,
        messages: Vec<ChatMessage>,
        image_urls: Vec<String>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<(String, usize, usize), MlxError> {
        let model_id = self.current_model().await.ok_or(MlxError::NotLoaded)?;
        let max_tokens = max_tokens.min(MAX_TOKEN_LIMIT).max(1);
        let text_prompt = messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content.as_text()))
            .collect::<Vec<_>>()
            .join("\n");

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(String, usize, usize)> {
                let mlx_vlm = py.import("mlx_vlm").map_err(|e| {
                    pyo3::exceptions::PyImportError::new_err(format!(
                        "mlx_vlm not installed (pip install mlx-vlm): {}", e
                    ))
                })?;

                let load_result = mlx_vlm.getattr("load")?.call1((&model_id,))?;
                let model = load_result.get_item(0)?;
                let processor = load_result.get_item(1)?;

                // Load images via PIL
                let pil = py.import("PIL.Image")?;
                let io_mod = py.import("io")?;
                let base64_mod = py.import("base64")?;
                let urllib = py.import("urllib.request")?;

                let mut pil_images: Vec<&PyAny> = Vec::new();
                for url in &image_urls {
                    let img = if url.starts_with("data:") {
                        let parts: Vec<&str> = url.splitn(2, ',').collect();
                        let data = parts.get(1).unwrap_or(&"");
                        let img_bytes = base64_mod.call_method1("b64decode", (*data,))?;
                        let buf = io_mod.call_method1("BytesIO", (img_bytes,))?;
                        pil.call_method1("open", (buf,))?
                    } else {
                        let resp = urllib.call_method1("urlopen", (url.as_str(),))?;
                        let data = resp.call_method0("read")?;
                        let buf = io_mod.call_method1("BytesIO", (data,))?;
                        pil.call_method1("open", (buf,))?
                    };
                    pil_images.push(img);
                }

                // Apply vision chat template
                let prompt_utils = mlx_vlm.getattr("prompt_utils")?;
                let tpl_kwargs = PyDict::new(py);
                tpl_kwargs.set_item("num_images", pil_images.len())?;
                let formatted: String = prompt_utils
                    .call_method("apply_chat_template", (processor, &text_prompt), Some(tpl_kwargs))?
                    .extract()?;

                // Generate
                let gen_kwargs = PyDict::new(py);
                gen_kwargs.set_item("max_tokens", max_tokens as i64)?;
                gen_kwargs.set_item("temp", temperature)?;
                gen_kwargs.set_item("top_p", top_p)?;
                let image_arg = if pil_images.len() == 1 {
                    pil_images[0].into_py(py)
                } else {
                    pyo3::types::PyList::new(py, &pil_images).into()
                };
                let response: String = mlx_vlm
                    .call_method("generate", (model, processor, &formatted, image_arg), Some(gen_kwargs))?
                    .extract()?;

                Ok((response, 0, 0))
            })
        })
        .await
        .map_err(|e| MlxError::Internal(e.to_string()))?
        .map_err(|e: PyErr| {
            let msg = e.to_string();
            if msg.contains("mlx_vlm not installed") || msg.contains("No module named") {
                MlxError::VisionNotAvailable(
                    "mlx_vlm not installed. Run: pip install mlx-vlm".into()
                )
            } else {
                MlxError::Python(msg)
            }
        })
    }

    async fn get_model_refs(&self) -> Result<(Py<PyAny>, Py<PyAny>, Option<Py<PyAny>>), MlxError> {
        self.get_model_refs_for(None).await
    }

    async fn get_model_refs_for(
        &self,
        adapter_name: Option<&str>,
    ) -> Result<(Py<PyAny>, Py<PyAny>, Option<Py<PyAny>>), MlxError> {
        let guard = self.inner.lock().await;
        if let Some(name) = adapter_name {
            let a = guard.adapters.get(name).ok_or_else(|| MlxError::AdapterNotFound(name.to_string()))?;
            let (m, t) = Python::with_gil(|py| (a.model.clone_ref(py), a.tokenizer.clone_ref(py)));
            return Ok((m, t, None));
        }
        let loaded = guard.loaded.as_ref().ok_or(MlxError::NotLoaded)?;
        let (m, t, d) = Python::with_gil(|py| {
            (
                loaded.model.clone_ref(py),
                loaded.tokenizer.clone_ref(py),
                loaded.draft_model.as_ref().map(|d| d.clone_ref(py)),
            )
        });
        Ok((m, t, d))
    }

    pub async fn mount_adapter(
        &self,
        name: String,
        model_id: String,
        adapter_path: String,
    ) -> Result<(), MlxError> {
        {
            let guard = self.inner.lock().await;
            if guard.adapters.len() >= MAX_MOUNTED_ADAPTERS && !guard.adapters.contains_key(&name) {
                return Err(MlxError::LoadFailed(format!(
                    "Adapter limit ({}) reached; unmount one first",
                    MAX_MOUNTED_ADAPTERS
                )));
            }
        }

        let mid = model_id.clone();
        let apath = adapter_path.clone();
        info!("Mounting adapter '{}' on model '{}' from '{}'", name, mid, apath);

        let (model_py, tokenizer_py) = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(PyObject, PyObject)> {
                let mlx_lm = py.import("mlx_lm")?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("adapter_path", apath.as_str())?;
                let result = mlx_lm.getattr("load")?.call((&mid,), Some(kwargs))?;
                let model: PyObject = result.get_item(0)?.into();
                let tokenizer: PyObject = result.get_item(1)?.into();
                Ok((model, tokenizer))
            })
        })
        .await
        .map_err(|e| MlxError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| MlxError::LoadFailed(e.to_string()))?;

        let mounted_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut guard = self.inner.lock().await;
        guard.adapters.insert(name.clone(), MountedAdapter {
            model: model_py,
            tokenizer: tokenizer_py,
            base_model_id: model_id,
            adapter_path,
            mounted_at,
        });
        info!("Adapter '{}' mounted", name);
        Ok(())
    }

    pub async fn unmount_adapter(&self, name: &str) -> bool {
        let mut guard = self.inner.lock().await;
        guard.adapters.remove(name).is_some()
    }

    pub async fn list_adapters(&self) -> Vec<MountedAdapterInfo> {
        let guard = self.inner.lock().await;
        guard.adapters.iter().map(|(name, a)| MountedAdapterInfo {
            name: name.clone(),
            base_model: a.base_model_id.clone(),
            adapter_path: a.adapter_path.clone(),
            mounted_at: a.mounted_at,
        }).collect()
    }

    pub fn new_chat_id() -> String {
        format!("chatcmpl-{}", &Uuid::new_v4().to_string().replace('-', "")[..12])
    }

    pub fn new_msg_id() -> String {
        format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24])
    }
}

// ── Pure helpers ─────────────────────────────────────────────────────────────

pub fn process_rss_mb() -> f64 {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|rss| rss as f64 / 1024.0)
        .unwrap_or(0.0)
}

fn prepare_messages(messages: &[ChatMessage]) -> Vec<HashMap<String, String>> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut rest: Vec<HashMap<String, String>> = Vec::new();

    for msg in messages {
        if msg.role == "system" {
            system_parts.push(msg.content.as_text());
        } else {
            let mut m = HashMap::new();
            m.insert("role".into(), msg.role.clone());
            m.insert("content".into(), msg.content.as_text());
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
        .map(|m| count_tokens(py, tokenizer, &m.content.as_text()))
        .sum();

    if total <= max_tokens {
        let mut result = system;
        result.extend(non_system);
        return result;
    }

    let mut kept: Vec<ChatMessage> = Vec::new();
    let mut used = 0usize;
    for msg in non_system.iter().rev() {
        let t = count_tokens(py, tokenizer, &msg.content.as_text());
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

pub fn available_ram_gb() -> Option<f64> {
    let output = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;

    let mut page_size = 4096u64;
    let mut free = 0u64;
    let mut inactive = 0u64;
    let mut purgeable = 0u64;
    let mut speculative = 0u64;

    for line in text.lines() {
        if line.contains("page size of") {
            if let Some(s) = line.split_whitespace()
                .find(|w| w.chars().all(|c| c.is_ascii_digit()))
                .and_then(|w| w.parse::<u64>().ok())
            {
                page_size = s;
            }
        } else if line.starts_with("Pages free:") {
            free = parse_vm_stat_line(line).unwrap_or(0);
        } else if line.starts_with("Pages inactive:") {
            inactive = parse_vm_stat_line(line).unwrap_or(0);
        } else if line.starts_with("Pages purgeable:") {
            purgeable = parse_vm_stat_line(line).unwrap_or(0);
        } else if line.starts_with("Pages speculative:") {
            speculative = parse_vm_stat_line(line).unwrap_or(0);
        }
    }

    let available_bytes = (free + inactive + purgeable + speculative) * page_size;
    Some(available_bytes as f64 / 1_073_741_824.0)
}

fn parse_vm_stat_line(line: &str) -> Option<u64> {
    line.split_whitespace()
        .last()?
        .trim_end_matches('.')
        .parse()
        .ok()
}

async fn fetch_model_size_gb(model_id: &str) -> Option<f64> {
    let url = format!("https://huggingface.co/api/models/{}", model_id);
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    resp.get("usedStorage")
        .and_then(|v| v.as_f64())
        .map(|b| b / 1_073_741_824.0)
}
