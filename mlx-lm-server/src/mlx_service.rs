use crate::config::Config;
use crate::error::MlxError;
use crate::models::{ChatMessage, MountedAdapterInfo, SamplerParams, Tool, ToolCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex as StdMutex};
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

struct PromptSession {
    kv_cache: Py<PyAny>, // Python List[KVCache]
    cached_tokens: usize,
}

pub struct PrefixEntry {
    pub snapshot: Py<PyAny>,  // deep-copied KV cache frozen at prefix boundary
    pub token_count: usize,   // tokens in the snapshot
    pub last_hit: u64,        // unix secs for LRU eviction
    pub hits: u64,
}

struct Inner {
    loaded: Option<LoadedModel>,
    adapters: HashMap<String, MountedAdapter>,
    prompt_sessions: HashMap<String, PromptSession>,
}

#[derive(Clone)]
pub struct MlxService {
    inner: Arc<Mutex<Inner>>,
    /// Prefix cache uses std::sync::Mutex so spawn_blocking closures can write to it directly.
    pub prefix_cache: Arc<StdMutex<HashMap<u64, PrefixEntry>>>,
    config: Arc<Config>,
}

impl MlxService {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { loaded: None, adapters: HashMap::new(), prompt_sessions: HashMap::new() })),
            prefix_cache: Arc::new(StdMutex::new(HashMap::new())),
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
        let moe_top_k = self.config.moe_top_k;
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

                // Apply MoE top-k override: reduces router fan-out for ~7-16% decode speedup
                if let Some(top_k) = moe_top_k {
                    let model_ref = model.as_ref(py);
                    apply_moe_top_k(py, model_ref, top_k)?;
                }

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

        // Clear prefix cache when model changes (cached KV blocks are model-specific)
        self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner()).clear();

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

    /// Retrieve existing session cache (if any) or create a new empty one.
    async fn get_or_create_session_cache(
        &self,
        session_id: &str,
        model_py: Py<PyAny>,
    ) -> (Py<PyAny>, usize) {
        {
            let guard = self.inner.lock().await;
            if let Some(s) = guard.prompt_sessions.get(session_id) {
                let cache_clone = Python::with_gil(|py| s.kv_cache.clone_ref(py));
                return (cache_clone, s.cached_tokens);
            }
        }
        // Create a new empty KV cache for this session
        let cache_opt = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Py<PyAny>> {
                let cache_mod = py.import("mlx_lm.models.cache")?;
                let kv = cache_mod.call_method1("make_prompt_cache", (model_py.as_ref(py),))?;
                Ok(kv.into())
            })
        })
        .await;
        match cache_opt {
            Ok(Ok(cache)) => (cache, 0),
            _ => {
                // Fallback: return a dummy that will be ignored
                Python::with_gil(|py| (py.None(), 0))
            }
        }
    }

    /// Store updated session after generation (offset read from Python cache object).
    pub async fn commit_session(&self, session_id: String, kv_cache: Py<PyAny>) {
        let cache_for_read = Python::with_gil(|py| kv_cache.clone_ref(py));
        let offset = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> usize {
                let locals = pyo3::types::PyDict::new(py);
                let _ = locals.set_item("cache", cache_for_read.as_ref(py));
                py.eval("cache[0].offset", None, Some(&locals))
                    .and_then(|v| v.extract::<usize>())
                    .unwrap_or(0)
            })
        })
        .await
        .unwrap_or(0);
        let mut guard = self.inner.lock().await;
        guard.prompt_sessions.insert(session_id, PromptSession { kv_cache, cached_tokens: offset });
    }

    pub async fn delete_prompt_session(&self, session_id: &str) {
        self.inner.lock().await.prompt_sessions.remove(session_id);
    }

    // ── Automatic Prefix Caching ──────────────────────────────────────────────

    /// Try to find a matching prefix snapshot in the APC cache and return a deep copy.
    pub async fn try_prefix_cache_hit(&self, prefix_msgs: &[ChatMessage]) -> Option<(Py<PyAny>, usize)> {
        if prefix_msgs.is_empty() {
            return None;
        }
        let hash = hash_messages(prefix_msgs);
        let (snapshot_ref, token_count) = {
            let mut cache = self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
            let entry = cache.get_mut(&hash)?;
            entry.last_hit = unix_secs();
            entry.hits += 1;
            let r = Python::with_gil(|py| entry.snapshot.clone_ref(py));
            (r, entry.token_count)
        };
        // Deep-copy the snapshot so each request gets its own mutable copy
        let deep_copy = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Py<PyAny>> {
                let copy_mod = py.import("copy")?;
                let deep = copy_mod.call_method1("deepcopy", (snapshot_ref.as_ref(py),))?;
                Ok(deep.into_py(py))
            })
        })
        .await
        .ok()?
        .ok()?;
        info!("APC: cache hit ({} tokens)", token_count);
        Some((deep_copy, token_count))
    }

    /// Build a KV snapshot for the given prefix messages and store it in the APC cache.
    /// Runs Python in the background; does not block inference.
    pub async fn build_and_store_prefix_snapshot(
        &self,
        prefix_msgs: Vec<ChatMessage>,
        model_py: Py<PyAny>,
        tokenizer_py: Py<PyAny>,
        chat_template_kwargs: serde_json::Value,
    ) {
        if prefix_msgs.is_empty() {
            return;
        }
        let hash = hash_messages(&prefix_msgs);
        if self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&hash) {
            return;
        }
        let apc_max = self.config.apc_max_entries;
        let prefix_cache = Arc::clone(&self.prefix_cache);

        tokio::task::spawn_blocking(move || {
            let result = Python::with_gil(|py| -> PyResult<(Py<PyAny>, usize)> {
                let _ = py.import("mlx.core").and_then(|mx| mx.call_method0("synchronize"));
                let tokenizer = tokenizer_py.as_ref(py);

                // Format prefix with add_generation_prompt=false to count exact prefix tokens
                let prefix_prompt = format_prefix_only(py, tokenizer, &prefix_msgs)?;
                let prefix_token_count = count_tokens(py, tokenizer, &prefix_prompt);
                if prefix_token_count == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("empty prefix prompt"));
                }

                // Create an empty KV cache and fill it with the prefix tokens via 1-token generation
                let cache_mod = py.import("mlx_lm.models.cache")?;
                let new_cache = cache_mod.call_method1("make_prompt_cache", (model_py.as_ref(py),))?;
                let new_cache: Py<PyAny> = new_cache.into_py(py);

                // Tokenize prefix to an mlx array (required format when prompt_cache is set)
                let mx = py.import("mlx.core")?;
                let all_ids: Vec<i64> = tokenizer.call_method1("encode", (&prefix_prompt,))?.extract()?;
                let prompt_arr = mx.call_method1("array", (all_ids,))?;

                let kwargs = PyDict::new(py);
                kwargs.set_item("max_tokens", 1i64)?;
                kwargs.set_item("verbose", false)?;
                kwargs.set_item("prompt", prompt_arr)?;
                kwargs.set_item("prompt_cache", new_cache.as_ref(py))?;

                let mlx_lm = py.import("mlx_lm")?;
                mlx_lm.call_method("generate", (model_py.as_ref(py), tokenizer), Some(kwargs))?;

                // Trim back to prefix_token_count (removes the 1 generated output token)
                let _ = cache_mod.call_method1(
                    "trim_prompt_cache",
                    (new_cache.as_ref(py), prefix_token_count as i64),
                );

                // Deep-copy to create an immutable snapshot independent of future mutations
                let copy_mod = py.import("copy")?;
                let snapshot = copy_mod.call_method1("deepcopy", (new_cache.as_ref(py),))?;
                Ok((snapshot.into_py(py), prefix_token_count))
            });

            match result {
                Ok((snapshot, token_count)) => {
                    let mut cache = prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
                    // LRU eviction when at capacity
                    if cache.len() >= apc_max && !cache.contains_key(&hash) {
                        if let Some(oldest) = cache.iter().min_by_key(|(_, e)| e.last_hit).map(|(&k, _)| k) {
                            cache.remove(&oldest);
                        }
                    }
                    cache.insert(hash, PrefixEntry { snapshot, token_count, last_hit: unix_secs(), hits: 0 });
                    info!("APC: stored prefix snapshot ({} tokens, {} entries)", token_count, cache.len());
                }
                Err(e) => warn!("APC: failed to build prefix snapshot: {}", e),
            }
        });
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

    pub async fn detokenize(&self, tokens: Vec<i64>) -> Result<String, MlxError> {
        let (_, tokenizer_py, _) = self.get_model_refs().await?;
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<String> {
                tokenizer_py.as_ref(py).call_method1("decode", (tokens,))?.extract()
            })
        })
        .await
        .map_err(|e| MlxError::Internal(e.to_string()))?
        .map_err(|e: PyErr| MlxError::Python(e.to_string()))
    }

    pub async fn apply_chat_template_prompt(
        &self,
        messages: Vec<crate::models::ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<(String, usize), MlxError> {
        let (_, tokenizer_py, _) = self.get_model_refs().await?;
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(String, usize)> {
                let tokenizer = tokenizer_py.as_ref(py);
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("add_generation_prompt", add_generation_prompt)?;
                let prompt = apply_chat_template(py, tokenizer, &messages, &serde_json::Value::Null, None)?;
                let token_count = count_tokens(py, tokenizer, &prompt);
                Ok((prompt, token_count))
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
        sampler: SamplerParams,
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
                let py_sampler = make_sampler(py, &sampler)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("prompt", prompt.as_str())?;
                kwargs.set_item("max_tokens", max_tokens as i64)?;
                kwargs.set_item("sampler", py_sampler)?;
                if let Some(bits) = kv_bits {
                    kwargs.set_item("kv_bits", bits)?;
                    kwargs.set_item("kv_group_size", kv_group_size.unwrap_or(64))?;
                }
                if let Some(v) = sampler.presence_penalty { kwargs.set_item("presence_penalty", v)?; }
                if let Some(v) = sampler.frequency_penalty { kwargs.set_item("frequency_penalty", v)?; }
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
        sampler: SamplerParams,
        chat_template_kwargs: serde_json::Value,
        kv_bits: Option<u32>,
        kv_group_size: Option<u32>,
        adapter_name: Option<String>,
        tools: Option<Vec<Tool>>,
        stop_strings: Vec<String>,
        want_logprobs: bool,
        top_n_logprobs: u32,
        seed: Option<u64>,
        session_id: Option<String>,
    ) -> Result<(String, usize, usize, Vec<ToolCall>, String, Option<Vec<serde_json::Value>>, Option<String>), MlxError> {
        let (model_py, tokenizer_py, draft_model_py) = self.get_model_refs_for(adapter_name.as_deref()).await?;

        // Extract prefix messages for APC before `messages` is moved into spawn_blocking
        let apc_prefix_msgs = if self.config.enable_apc && session_id.is_none() {
            extract_prefix_messages(&messages)
        } else {
            vec![]
        };

        // Get or create session KV cache; fall back to APC prefix cache when no session_id
        let session_cache = if let Some(ref sid) = session_id {
            let model_py_for_cache = Python::with_gil(|py| model_py.clone_ref(py));
            let (cache, offset) = self.get_or_create_session_cache(sid, model_py_for_cache).await;
            Some((cache, offset))
        } else if !apc_prefix_msgs.is_empty() {
            self.try_prefix_cache_hit(&apc_prefix_msgs).await
        } else {
            None
        };

        // Clone refs for APC snapshot building (happens after generation, before they're dropped)
        let (model_py_apc, tokenizer_py_apc) = if !apc_prefix_msgs.is_empty() && session_cache.is_none() {
            (
                Some(Python::with_gil(|py| model_py.clone_ref(py))),
                Some(Python::with_gil(|py| tokenizer_py.clone_ref(py))),
            )
        } else {
            (None, None)
        };

        // Clone handles for use inside spawn_blocking; original retained for post-generation commit
        let session_cache_for_spawn: Option<(Py<PyAny>, usize)> = session_cache.as_ref()
            .map(|(cache, offset)| (Python::with_gil(|py| cache.clone_ref(py)), *offset));
        // Clone chat_template_kwargs for APC use BEFORE it's moved into spawn_blocking
        let chat_template_kwargs_apc = chat_template_kwargs.clone();
        let max_tokens = max_tokens.min(MAX_TOKEN_LIMIT).max(1);

        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(String, usize, usize, Vec<ToolCall>, String, Option<Vec<serde_json::Value>>, Option<String>)> {
                // Ensure Metal stream is initialized on this thread before any GPU ops
                let _ = py.import("mlx.core").and_then(|mx| mx.call_method0("synchronize"));
                let tokenizer = tokenizer_py.as_ref(py);
                let windowed = apply_sliding_window(messages, MAX_WINDOW_TOKENS, py, tokenizer);
                let prompt = apply_chat_template(py, tokenizer, &windowed, &chat_template_kwargs, tools.as_deref())?;
                let prompt_tokens = count_tokens(py, tokenizer, &prompt);

                let mlx_lm = py.import("mlx_lm")?;
                let py_sampler = make_sampler(py, &sampler)?;

                // Apply seed if provided
                if let Some(s) = seed {
                    let _ = py.import("mlx.core").and_then(|mx| {
                        mx.getattr("random")?.call_method1("seed", (s as i64,))
                    });
                }

                let kwargs = PyDict::new(py);
                kwargs.set_item("max_tokens", max_tokens as i64)?;
                kwargs.set_item("sampler", py_sampler)?;
                if let Some(bits) = kv_bits {
                    kwargs.set_item("kv_bits", bits)?;
                    kwargs.set_item("kv_group_size", kv_group_size.unwrap_or(64))?;
                }
                if let Some(ref draft) = draft_model_py {
                    kwargs.set_item("draft_model", draft.as_ref(py))?;
                    if let Some(n) = sampler.num_draft_tokens {
                        kwargs.set_item("num_draft_tokens", n as i64)?;
                    }
                }
                if let Some(v) = sampler.presence_penalty { kwargs.set_item("presence_penalty", v)?; }
                if let Some(v) = sampler.frequency_penalty { kwargs.set_item("frequency_penalty", v)?; }

                // Logits processors: collect all, then set once
                let mut processors_list: Vec<PyObject> = Vec::new();
                if let Some(budget) = sampler.thinking_budget {
                    if let Ok(proc) = make_thinking_budget_processor(py, tokenizer, budget) {
                        processors_list.push(proc.into_py(py));
                    }
                }
                if let Some(ref bias_map) = sampler.logit_bias {
                    if !bias_map.is_empty() {
                        if let Ok(proc) = make_logit_bias_processor(py, bias_map) {
                            processors_list.push(proc.into_py(py));
                        }
                    }
                }
                if !processors_list.is_empty() {
                    let processors = PyList::new(py, &processors_list);
                    kwargs.set_item("logits_processors", processors)?;
                }

                // Set up prompt and optional KV cache
                if let Some((ref kv_cache, cached_offset)) = session_cache_for_spawn {
                    // Tokenize full prompt to find delta beyond cached tokens
                    let mx = py.import("mlx.core")?;
                    let all_ids: Vec<i64> = tokenizer.call_method1("encode", (&prompt,))?.extract()?;
                    let delta_ids = if cached_offset > 0 && cached_offset < all_ids.len() {
                        all_ids[cached_offset..].to_vec()
                    } else {
                        all_ids
                    };
                    let delta_arr = mx.call_method1("array", (delta_ids,))?;
                    kwargs.set_item("prompt", delta_arr)?;
                    kwargs.set_item("prompt_cache", kv_cache.as_ref(py))?;
                } else {
                    kwargs.set_item("prompt", &prompt)?;
                }

                let (mut response, mut finish_reason, mut lp_list) = if want_logprobs {
                    // Use stream_generate to collect per-token logprobs
                    let generator = mlx_lm
                        .getattr("stream_generate")?
                        .call((model_py.as_ref(py), tokenizer), Some(kwargs))?;
                    let mut text = String::new();
                    let mut lps: Vec<serde_json::Value> = Vec::new();
                    let mut fr = "stop".to_string();
                    for item in generator.iter()? {
                        let r = item?;
                        let tok_text: String = r.getattr("text")?.extract()?;
                        let tok_id: i64 = r.getattr("token")?.extract()?;
                        let logprobs_arr = r.getattr("logprobs")?;
                        let tok_lp: f32 = logprobs_arr.get_item(tok_id)?.extract()?;
                        text.push_str(&tok_text);

                        // Build top-N logprobs using Python-side numpy-style operations
                        let top_lps = if top_n_logprobs > 0 {
                            collect_top_logprobs(py, tokenizer, logprobs_arr.into(), top_n_logprobs)
                        } else {
                            vec![]
                        };

                        lps.push(serde_json::json!({
                            "token": tok_text,
                            "logprob": tok_lp,
                            "top_logprobs": top_lps,
                        }));
                        if let Ok(reason_obj) = r.getattr("finish_reason") {
                            if !reason_obj.is_none() {
                                if let Ok(reason_str) = reason_obj.extract::<String>() {
                                    fr = reason_str;
                                }
                            }
                        }
                    }
                    (text, fr, Some(lps))
                } else if let Some(ref grammar_str) = sampler.grammar {
                    let response = constrained_generate(py, model_py.as_ref(py), tokenizer, &prompt, grammar_str, max_tokens)
                        .unwrap_or_else(|_| {
                            // Fallback: unconstrained if outlines not available
                            mlx_lm.getattr("generate")
                                .and_then(|g| g.call((model_py.as_ref(py), tokenizer), Some(kwargs)))
                                .and_then(|r| r.extract::<String>())
                                .unwrap_or_default()
                        });
                    let fr = if count_tokens(py, tokenizer, &response) >= max_tokens {
                        "length".to_string()
                    } else {
                        "stop".to_string()
                    };
                    (response, fr, None)
                } else {
                    let response: String = mlx_lm
                        .getattr("generate")?
                        .call((model_py.as_ref(py), tokenizer), Some(kwargs))?
                        .extract()?;
                    let fr = if count_tokens(py, tokenizer, &response) >= max_tokens {
                        "length".to_string()
                    } else {
                        "stop".to_string()
                    };
                    (response, fr, None)
                };

                // Apply stop strings: truncate at first occurrence
                for stop in &stop_strings {
                    if let Some(idx) = response.find(stop.as_str()) {
                        response.truncate(idx);
                        finish_reason = "stop".to_string();
                        // Trim logprobs list to match truncated response
                        if let Some(ref mut lps) = lp_list {
                            let kept_chars = response.chars().count();
                            let _ = kept_chars; // logprob count may differ; best-effort trim
                        }
                        break;
                    }
                }

                let completion_tokens = count_tokens(py, tokenizer, &response);
                let parsed_tool_calls = if tools.is_some() {
                    parse_tool_calls(py, tokenizer, &response).unwrap_or_default()
                } else {
                    vec![]
                };
                if !parsed_tool_calls.is_empty() {
                    finish_reason = "tool_calls".to_string();
                }
                let (response, reasoning) = extract_reasoning(response);
                Ok((response, prompt_tokens, completion_tokens, parsed_tool_calls, finish_reason, lp_list, reasoning))
            })
        })
        .await
        .map_err(|e| MlxError::Internal(e.to_string()))?
        .map_err(|e: PyErr| MlxError::Python(e.to_string()))?;

        // Update session KV cache (in-place mutation happened in Python)
        if let (Some(sid), Some((kv_cache, _))) = (session_id, session_cache) {
            self.commit_session(sid, kv_cache).await;
        } else if let (Some(model_ref), Some(tok_ref)) = (model_py_apc, tokenizer_py_apc) {
            // APC: build prefix snapshot in background so future requests benefit
            let service = self.clone();
            let prefix = apc_prefix_msgs;
            tokio::spawn(async move {
                service.build_and_store_prefix_snapshot(prefix, model_ref, tok_ref, chat_template_kwargs_apc).await;
            });
        }

        Ok(result)
    }

    pub async fn generate_stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        sampler: SamplerParams,
        timeout_secs: f64,
        chat_template_kwargs: serde_json::Value,
        kv_bits: Option<u32>,
        kv_group_size: Option<u32>,
        adapter_name: Option<String>,
        tools: Option<Vec<Tool>>,
        stop_strings: Vec<String>,
        want_logprobs: bool,
        top_n_logprobs: u32,
        seed: Option<u64>,
        session_id: Option<String>,
    ) -> Result<(ReceiverStream<Result<String, MlxError>>, Option<(String, Py<PyAny>, usize)>), MlxError> {
        let (model_py, tokenizer_py, draft_model_py) = self.get_model_refs_for(adapter_name.as_deref()).await?;

        // Extract prefix messages for APC before `messages` is moved into spawn_blocking
        let apc_prefix_msgs = if self.config.enable_apc && session_id.is_none() {
            extract_prefix_messages(&messages)
        } else {
            vec![]
        };

        // Get or create session KV cache; fall back to APC prefix cache when no session_id
        let session_cache = if let Some(ref sid) = session_id {
            let model_py_for_cache = Python::with_gil(|py| model_py.clone_ref(py));
            let (cache, offset) = self.get_or_create_session_cache(sid, model_py_for_cache).await;
            Some((cache, offset))
        } else if !apc_prefix_msgs.is_empty() {
            self.try_prefix_cache_hit(&apc_prefix_msgs).await
        } else {
            None
        };

        // Clone refs for background APC snapshot building after generation starts
        let (model_py_apc, tokenizer_py_apc) = if !apc_prefix_msgs.is_empty() && session_cache.is_none() {
            (
                Some(Python::with_gil(|py| model_py.clone_ref(py))),
                Some(Python::with_gil(|py| tokenizer_py.clone_ref(py))),
            )
        } else {
            (None, None)
        };

        // Clone handles for use inside spawn_blocking; original retained for session_info return
        let session_cache_for_spawn: Option<(Py<PyAny>, usize)> = session_cache.as_ref()
            .map(|(cache, offset)| (Python::with_gil(|py| cache.clone_ref(py)), *offset));
        // Clone chat_template_kwargs for APC use BEFORE it's moved into spawn_blocking
        let chat_template_kwargs_apc = chat_template_kwargs.clone();
        let max_tokens = max_tokens.min(MAX_TOKEN_LIMIT).max(1);
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, MlxError>>(64);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let run = || -> PyResult<()> {
                    // Ensure Metal stream is initialized on this thread before any GPU ops
                    let _ = py.import("mlx.core").and_then(|mx| mx.call_method0("synchronize"));
                    let tokenizer = tokenizer_py.as_ref(py);
                    let windowed = apply_sliding_window(messages, MAX_WINDOW_TOKENS, py, tokenizer);
                    let prompt =
                        apply_chat_template(py, tokenizer, &windowed, &chat_template_kwargs, tools.as_deref())?;

                    let mlx_lm = py.import("mlx_lm")?;
                    let py_sampler = make_sampler(py, &sampler)?;

                    // Apply seed before generation if provided
                    if let Some(s) = seed {
                        let _ = py.import("mlx.core").and_then(|mx| {
                            mx.getattr("random")?.call_method1("seed", (s as i64,))
                        });
                    }

                    let kwargs = PyDict::new(py);
                    kwargs.set_item("max_tokens", max_tokens as i64)?;
                    kwargs.set_item("sampler", py_sampler)?;
                    if let Some(bits) = kv_bits {
                        kwargs.set_item("kv_bits", bits)?;
                        kwargs.set_item("kv_group_size", kv_group_size.unwrap_or(64))?;
                    }
                    if let Some(ref draft) = draft_model_py {
                        kwargs.set_item("draft_model", draft.as_ref(py))?;
                        if let Some(n) = sampler.num_draft_tokens {
                            kwargs.set_item("num_draft_tokens", n as i64)?;
                        }
                    }
                    if let Some(v) = sampler.presence_penalty { kwargs.set_item("presence_penalty", v)?; }
                    if let Some(v) = sampler.frequency_penalty { kwargs.set_item("frequency_penalty", v)?; }

                    // Logits processors for streaming
                    let mut stream_procs: Vec<PyObject> = Vec::new();
                    if let Some(budget) = sampler.thinking_budget {
                        if let Ok(proc) = make_thinking_budget_processor(py, tokenizer, budget) {
                            stream_procs.push(proc.into_py(py));
                        }
                    }
                    if let Some(ref bias_map) = sampler.logit_bias {
                        if !bias_map.is_empty() {
                            if let Ok(proc) = make_logit_bias_processor(py, bias_map) {
                                stream_procs.push(proc.into_py(py));
                            }
                        }
                    }
                    if !stream_procs.is_empty() {
                        let processors = PyList::new(py, &stream_procs);
                        kwargs.set_item("logits_processors", processors)?;
                    }

                    // Set up prompt with optional session KV cache
                    if let Some((ref kv_cache, cached_offset)) = session_cache_for_spawn {
                        let mx = py.import("mlx.core")?;
                        let all_ids: Vec<i64> = tokenizer.call_method1("encode", (&prompt,))?.extract()?;
                        let delta_ids = if cached_offset > 0 && cached_offset < all_ids.len() {
                            all_ids[cached_offset..].to_vec()
                        } else {
                            all_ids
                        };
                        let delta_arr = mx.call_method1("array", (delta_ids,))?;
                        kwargs.set_item("prompt", delta_arr)?;
                        kwargs.set_item("prompt_cache", kv_cache.as_ref(py))?;
                    } else {
                        kwargs.set_item("prompt", &prompt)?;
                    }

                    let generator = mlx_lm
                        .getattr("stream_generate")?
                        .call((model_py.as_ref(py), tokenizer), Some(kwargs))?;

                    // When tools are provided, buffer full output to parse tool calls at end.
                    let has_tools = tools.is_some();
                    let mut full_output: Option<String> = if has_tools { Some(String::new()) } else { None };
                    // For stop string detection we need to track accumulated text.
                    // We keep a rolling tail of len = max stop string length to catch
                    // stop strings that span token boundaries.
                    let max_stop_len = stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);
                    let mut accumulated = String::new();
                    let mut emitted_len = 0usize; // bytes of accumulated already sent
                    let mut finish_reason = "stop".to_string();
                    let mut stop_hit = false;

                    let start = Instant::now();
                    'gen: for item in generator.iter()? {
                        if start.elapsed().as_secs_f64() > timeout_secs {
                            warn!("Stream timeout exceeded");
                            let _ = tx.blocking_send(Err(MlxError::Timeout));
                            return Ok(());
                        }
                        let response = item?;
                        let text: String = response.getattr("text")?.extract()?;
                        let tok_id: i64 = response.getattr("token")?.extract()?;
                        let logprobs_arr = response.getattr("logprobs")?;
                        let fr: Option<String> = response.getattr("finish_reason")
                            .and_then(|v| if v.is_none() { Err(pyo3::exceptions::PyAttributeError::new_err("")) } else { v.extract() })
                            .ok();
                        if let Some(r) = fr {
                            finish_reason = r;
                        }

                        if let Some(ref mut buf) = full_output {
                            buf.push_str(&text);
                            continue;
                        }

                        // Build logprob sentinel if requested (only when no stop buffering is active)
                        let lp_sentinel = if want_logprobs && stop_strings.is_empty() {
                            let tok_lp: f32 = logprobs_arr.get_item(tok_id).and_then(|v| v.extract()).unwrap_or(0.0);
                            let top_lps = collect_top_logprobs(py, tokenizer, logprobs_arr.into(), top_n_logprobs);
                            let lp_json = serde_json::json!({
                                "token": &text,
                                "logprob": tok_lp,
                                "top_logprobs": top_lps,
                            });
                            Some(format!("\x00LP:{}\x00", lp_json))
                        } else {
                            None
                        };

                        // Stop string detection on accumulated text
                        if !stop_strings.is_empty() {
                            accumulated.push_str(&text);
                            for stop in &stop_strings {
                                if let Some(idx) = accumulated.find(stop.as_str()) {
                                    // Emit everything before the stop string that hasn't been sent
                                    if idx > emitted_len {
                                        let to_emit = accumulated[emitted_len..idx].to_string();
                                        if !to_emit.is_empty() {
                                            let _ = tx.blocking_send(Ok(to_emit));
                                        }
                                    }
                                    finish_reason = "stop".to_string();
                                    stop_hit = true;
                                    break 'gen;
                                }
                            }
                            // Safe to emit up to (accumulated.len() - max_stop_len) bytes
                            // to avoid emitting part of a potential future stop string
                            let safe_end = accumulated.len().saturating_sub(max_stop_len);
                            if safe_end > emitted_len {
                                let to_emit = accumulated[emitted_len..safe_end].to_string();
                                emitted_len = safe_end;
                                if !to_emit.is_empty() && tx.blocking_send(Ok(to_emit)).is_err() {
                                    break;
                                }
                            }
                        } else {
                            // Emit logprob sentinel first so handler can attach it to this token
                            if let Some(lp_s) = lp_sentinel {
                                let _ = tx.blocking_send(Ok(lp_s));
                            }
                            if tx.blocking_send(Ok(text)).is_err() {
                                break;
                            }
                        }
                    }

                    // Flush any remaining buffered text (stop strings path, only if no stop was hit)
                    if !stop_strings.is_empty() && full_output.is_none() && !stop_hit && emitted_len < accumulated.len() {
                        let remainder = accumulated[emitted_len..].to_string();
                        if !remainder.is_empty() {
                            let _ = tx.blocking_send(Ok(remainder));
                        }
                    }

                    // Tool calls path: parse and emit sentinel
                    if let Some(buf) = full_output {
                        let tool_calls = parse_tool_calls(py, tokenizer, &buf).unwrap_or_default();
                        if !tool_calls.is_empty() {
                            let sentinel = format!(
                                "\x00TOOL_CALLS:{}\x00",
                                serde_json::to_string(&tool_calls).unwrap_or_default()
                            );
                            let _ = tx.blocking_send(Ok(sentinel));
                            finish_reason = "tool_calls".to_string();
                        } else {
                            let _ = tx.blocking_send(Ok(buf));
                        }
                    }

                    // Emit finish_reason sentinel for the stream handler
                    let _ = tx.blocking_send(Ok(format!("\x00FINISH:{}\x00", finish_reason)));
                    Ok(())
                };

                if let Err(e) = run() {
                    error!("Streaming Python error: {}", e);
                    let _ = tx.blocking_send(Err(MlxError::Python(e.to_string())));
                }
            });
        });

        // Kick off background APC snapshot building (first request with this prefix pays a small
        // background cost; subsequent requests get a free KV cache fast-path)
        if let (Some(model_ref), Some(tok_ref)) = (model_py_apc, tokenizer_py_apc) {
            let service = self.clone();
            let prefix = apc_prefix_msgs;
            tokio::spawn(async move {
                service.build_and_store_prefix_snapshot(prefix, model_ref, tok_ref, chat_template_kwargs_apc).await;
            });
        }

        // Return stream + session info for the caller to commit after stream ends
        let session_info = session_id.zip(session_cache).map(|(sid, (cache, _))| (sid, cache, 0usize));
        Ok((ReceiverStream::new(rx), session_info))
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

    /// Fire max_tokens=1 completions for each warm-up message set to prime the prefix KV cache.
    pub async fn warm_up(&self, message_sets: Vec<Vec<crate::models::ChatMessage>>) {
        for messages in message_sets {
            let sampler = crate::models::SamplerParams {
                temperature: 0.0,
                top_p: 1.0,
                ..Default::default()
            };
            let _ = self.generate_response(
                messages,
                1,
                sampler,
                serde_json::Value::Object(Default::default()),
                None, None, None, None, vec![], false, 0, None, None,
            ).await;
        }
        info!("Warm-up complete");
    }

    /// Score query against documents using bi-encoder cosine similarity.
    /// Returns scores in the same order as input documents.
    pub async fn rerank(&self, query: String, documents: Vec<String>) -> Result<Vec<f32>, MlxError> {
        let all_texts: Vec<String> = std::iter::once(query)
            .chain(documents.into_iter())
            .collect();
        let embeddings = self.get_embeddings(all_texts).await?;
        let mut iter = embeddings.into_iter();
        let query_emb = iter.next().unwrap_or_default();
        let scores = iter.map(|doc_emb| cosine_similarity(&query_emb, &doc_emb)).collect();
        Ok(scores)
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

fn parse_tool_calls(py: Python, tokenizer: &PyAny, text: &str) -> PyResult<Vec<ToolCall>> {
    // Detect which tool parser the tokenizer uses (if any)
    let has_tool_calling = tokenizer
        .getattr("has_tool_calling")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);

    if !has_tool_calling {
        // Fallback: try json_tools parser directly (many models use <tool_call>...</tool_call>)
        let has_delimiters = text.contains("<tool_call>") && text.contains("</tool_call>");
        if !has_delimiters {
            return Ok(vec![]);
        }
    }

    let tool_parser = match tokenizer.getattr("tool_parser") {
        Ok(p) if !p.is_none() => p,
        _ => {
            // Try json_tools as fallback
            let m = py.import("mlx_lm.tool_parsers.json_tools")?;
            m.getattr("parse_tool_call")?
        }
    };

    let tool_call_start: String = tokenizer
        .getattr("tool_call_start")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| "<tool_call>".into());
    let tool_call_end: String = tokenizer
        .getattr("tool_call_end")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| "</tool_call>".into());

    // Extract all segments between delimiters
    let mut tool_calls = vec![];
    let mut remaining = text;
    while let Some(start) = remaining.find(tool_call_start.as_str()) {
        let after_start = &remaining[start + tool_call_start.len()..];
        let end = if tool_call_end.is_empty() {
            after_start.len()
        } else {
            after_start.find(tool_call_end.as_str()).unwrap_or(after_start.len())
        };
        let segment = &after_start[..end];
        match tool_parser.call1((segment,)) {
            Ok(parsed) => {
                let name: String = parsed.get_item("name")?.extract()?;
                let args_py = parsed.get_item("arguments")?;
                let args_val: serde_json::Value = if let Ok(s) = args_py.extract::<String>() {
                    serde_json::from_str(&s).unwrap_or(serde_json::Value::Object(Default::default()))
                } else {
                    let args_str: String = py
                        .import("json")?
                        .call_method1("dumps", (args_py,))?
                        .extract()?;
                    serde_json::from_str(&args_str).unwrap_or(serde_json::Value::Object(Default::default()))
                };
                tool_calls.push(ToolCall::new(name, args_val));
            }
            Err(e) => {
                warn!("Failed to parse tool call segment: {}", e);
            }
        }
        remaining = if tool_call_end.is_empty() {
            ""
        } else {
            let skip = start + tool_call_start.len() + end + tool_call_end.len();
            &remaining[skip.min(remaining.len())..]
        };
    }
    Ok(tool_calls)
}

fn apply_chat_template(
    py: Python,
    tokenizer: &PyAny,
    messages: &[ChatMessage],
    extra_kwargs: &serde_json::Value,
    tools: Option<&[Tool]>,
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

        // Pass tools to the chat template when provided — models like Llama-3,
        // Qwen, and Mistral encode tool definitions into the system prompt via
        // their chat template's `tools` variable.
        if let Some(tool_list) = tools {
            let py_tools = PyList::new(
                py,
                tool_list.iter().map(|t| {
                    let d = PyDict::new(py);
                    d.set_item("type", &t.kind).unwrap();
                    let f = PyDict::new(py);
                    f.set_item("name", &t.function.name).unwrap();
                    if let Some(desc) = &t.function.description {
                        f.set_item("description", desc).unwrap();
                    }
                    if let Some(params) = &t.function.parameters {
                        if let Ok(py_params) = json_to_py(py, params) {
                            f.set_item("parameters", py_params).unwrap();
                        }
                    }
                    d.set_item("function", f).unwrap();
                    d
                }),
            );
            kwargs.set_item("tools", py_tools)?;
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

fn make_sampler<'py>(py: Python<'py>, p: &SamplerParams) -> PyResult<&'py PyAny> {
    // Mirostat v2 — replaces the default sampler entirely
    if p.mirostat == Some(true) {
        let tau = p.mirostat_tau.unwrap_or(5.0);
        let eta = p.mirostat_eta.unwrap_or(0.1);
        let temp = p.temperature;
        return make_mirostat_sampler(py, tau, eta, temp);
    }
    // Dynamic temperature — wraps the base sampler
    if let Some(dynatemp_range) = p.dynatemp_range {
        if dynatemp_range > 0.0 {
            let exponent = p.dynatemp_exponent.unwrap_or(1.0);
            let temp = p.temperature;
            return make_dynatemp_sampler(py, temp, dynatemp_range, exponent);
        }
    }
    // Default mlx-lm sampler
    let kwargs = PyDict::new(py);
    kwargs.set_item("temp", p.temperature)?;
    kwargs.set_item("top_p", p.top_p)?;
    if let Some(v) = p.top_k     { kwargs.set_item("top_k", v)?; }
    if let Some(v) = p.min_p     { kwargs.set_item("min_p", v)?; }
    if let Some(v) = p.repetition_penalty { kwargs.set_item("repetition_penalty", v)?; }
    if let Some(v) = p.xtc_probability { kwargs.set_item("xtc_probability", v)?; }
    if let Some(v) = p.xtc_threshold   { kwargs.set_item("xtc_threshold", v)?; }
    py.import("mlx_lm.sample_utils")?
        .getattr("make_sampler")?
        .call((), Some(kwargs))
}

fn make_mirostat_sampler<'py>(py: Python<'py>, tau: f64, eta: f64, temp: f64) -> PyResult<&'py PyAny> {
    let code = r#"
import mlx.core as _mx
import math as _math
import random as _random

class _MirostatV2Sampler:
    """Mirostat v2: adapts temperature per step to target perplexity tau."""
    def __init__(self, tau, eta, temp):
        self.tau = tau
        self.eta = eta
        self.temp = temp
        self.mu = 2.0 * tau  # running surprise estimate

    def __call__(self, logits):
        if self.temp > 0:
            logits = logits / self.temp
        # Work on top-2000 tokens to avoid full 128K vocab iteration
        k = min(2000, int(logits.shape[-1]))
        top_idx = _mx.argpartition(-logits, k)[:k].tolist()
        top_logits = logits[top_idx].tolist()
        max_l = max(top_logits)
        exps = [_math.exp(l - max_l) for l in top_logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        pairs = sorted(zip(top_idx, probs), key=lambda x: -x[1])
        # Mirostat truncation: drop tokens whose surprise exceeds mu
        kept = []
        for tok_id, p in pairs:
            if p <= 0:
                continue
            if -_math.log2(p) > self.mu and len(kept) > 0:
                break
            kept.append((tok_id, p))
        if not kept:
            kept = pairs[:1]
        # Normalize and sample
        kt = sum(p for _, p in kept)
        r = _random.random() * kt
        cum = 0.0
        selected, sel_p = kept[0]
        for tok_id, p in kept:
            cum += p
            if r <= cum:
                selected, sel_p = tok_id, p
                break
        # Update mu
        surprise = -_math.log2(max(sel_p, 1e-10))
        self.mu = self.mu - self.eta * (surprise - self.tau)
        return _mx.array(int(selected))

_mirostat_sampler = _MirostatV2Sampler(_tau, _eta, _temp)
"#;
    let locals = PyDict::new(py);
    locals.set_item("_tau", tau)?;
    locals.set_item("_eta", eta)?;
    locals.set_item("_temp", temp)?;
    py.run(code, None, Some(locals))?;
    Ok(locals.get_item("_mirostat_sampler")
        .and_then(|o| o.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("mirostat sampler not found")))?)
}

fn make_dynatemp_sampler<'py>(py: Python<'py>, temp: f64, range: f64, exponent: f64) -> PyResult<&'py PyAny> {
    let code = r#"
import mlx.core as _mx
import math as _math

class _DynaTempSampler:
    """Dynamic temperature: scales temperature by normalised entropy of the logit distribution."""
    def __init__(self, temp, dynatemp_range, dynatemp_exponent):
        self.temp = temp
        self.range = dynatemp_range
        self.exp = dynatemp_exponent

    def __call__(self, logits):
        # Compute entropy over top-1000 tokens for speed
        k = min(1000, int(logits.shape[-1]))
        top_idx = _mx.argpartition(-logits, k)[:k].tolist()
        top_l = logits[top_idx].tolist()
        max_l = max(top_l)
        exps = [_math.exp(l - max_l) for l in top_l]
        total = sum(exps)
        probs = [e / total for e in exps]
        entropy = -sum(p * _math.log(max(p, 1e-10)) for p in probs)
        max_entropy = _math.log(k)
        norm_entropy = entropy / max(max_entropy, 1e-10)
        # Scale temperature: high entropy → higher temp, low entropy → lower temp
        scale = 1.0 + self.range * (norm_entropy ** self.exp - 0.5)
        t = max(0.01, self.temp * scale)
        return _mx.random.categorical(logits / t)

_dynatemp_sampler = _DynaTempSampler(_temp, _range, _exp)
"#;
    let locals = PyDict::new(py);
    locals.set_item("_temp", temp)?;
    locals.set_item("_range", range)?;
    locals.set_item("_exp", exponent)?;
    py.run(code, None, Some(locals))?;
    Ok(locals.get_item("_dynatemp_sampler")
        .and_then(|o| o.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("dynatemp sampler not found")))?)
}

/// Walk model layers and reduce top_k on MoE router layers.
fn apply_moe_top_k(py: Python, model: &PyAny, top_k: u32) -> PyResult<()> {
    let top_k_i = top_k as i64;
    let count = apply_moe_top_k_recursive(py, model, top_k_i)?;
    if count > 0 {
        info!("MoE top_k override: set top_k={} on {} layer(s)", top_k, count);
    }
    Ok(())
}

fn apply_moe_top_k_recursive(py: Python, module: &PyAny, top_k: i64) -> PyResult<usize> {
    let mut count = 0usize;
    let cls_name = module.get_type().name()
        .ok()
        .and_then(|s| s.extract::<String>().ok())
        .unwrap_or_default()
        .to_lowercase();
    let is_moe = ["moe", "switch", "sparse", "expert"]
        .iter()
        .any(|kw| cls_name.contains(kw));
    if is_moe {
        if let Ok(current) = module.getattr("top_k") {
            if !current.is_none() {
                module.setattr("top_k", top_k)?;
                count += 1;
            }
        }
    }
    // Recurse into children via .children() iterator if available
    if let Ok(children_fn) = module.getattr("children") {
        if let Ok(children) = children_fn.call0() {
            if let Ok(iter) = children.iter() {
                for child in iter.flatten() {
                    count += apply_moe_top_k_recursive(py, child, top_k).unwrap_or(0);
                }
            }
        }
    }
    Ok(count)
}

/// Build a Python logits processor that forces </think> after `budget` reasoning tokens.
fn make_thinking_budget_processor<'py>(
    py: Python<'py>,
    tokenizer: &PyAny,
    budget: usize,
) -> PyResult<&'py PyAny> {
    let code = r#"
class _ThinkingBudgetProcessor:
    def __init__(self, tokenizer, budget):
        self.budget = budget
        self.think_count = 0
        self.in_think = False
        try:
            ids = tokenizer.encode("<think>", add_special_tokens=False)
            self.think_start_id = int(ids[-1]) if ids else None
        except Exception:
            self.think_start_id = None
        try:
            ids_end = tokenizer.encode("</think>", add_special_tokens=False)
            self.think_end_id = int(ids_end[-1]) if ids_end else None
        except Exception:
            self.think_end_id = None

    def __call__(self, tokens, logits):
        import numpy as _np
        import mlx.core as _mx
        if tokens.size > 0:
            last = int(tokens[-1])
            if self.think_start_id is not None and last == self.think_start_id:
                self.in_think = True
                self.think_count = 0
            elif self.think_end_id is not None and last == self.think_end_id:
                self.in_think = False
        if self.in_think:
            self.think_count += 1
            if self.think_count >= self.budget and self.think_end_id is not None:
                vocab_size = int(logits.shape[-1])
                forced = _np.full((vocab_size,), -1e9, dtype=_np.float32)
                forced[self.think_end_id] = 0.0
                return _mx.array(forced)
        return logits

_thinking_budget_proc = _ThinkingBudgetProcessor(_tokenizer, _budget)
"#;
    let locals = PyDict::new(py);
    locals.set_item("_tokenizer", tokenizer)?;
    locals.set_item("_budget", budget as i64)?;
    py.run(code, None, Some(locals))?;
    Ok(locals.get_item("_thinking_budget_proc")
        .and_then(|o| o.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("processor not found")))?)
}

fn make_logit_bias_processor<'py>(
    py: Python<'py>,
    bias_map: &std::collections::HashMap<String, f64>,
) -> PyResult<&'py PyAny> {
    let bias_json = serde_json::to_string(bias_map).unwrap_or_else(|_| "{}".into());
    let code = r#"
import json as _json
import mlx.core as _mx
import numpy as _np

class _LogitBiasProcessor:
    def __init__(self, bias_map):
        self.bias = {int(k): float(v) for k, v in bias_map.items()}

    def __call__(self, tokens, logits):
        if not self.bias:
            return logits
        arr = _np.array(logits.tolist(), dtype=_np.float32)
        for tok_id, bias_val in self.bias.items():
            if 0 <= tok_id < arr.shape[-1]:
                arr[tok_id] += bias_val
        return _mx.array(arr)

_logit_bias_proc = _LogitBiasProcessor(_json.loads(_bias_json))
"#;
    let locals = PyDict::new(py);
    locals.set_item("_bias_json", bias_json)?;
    py.run(code, None, Some(locals))?;
    Ok(locals.get_item("_logit_bias_proc")
        .and_then(|o| o.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("processor not found")))?)
}

/// Grammar-constrained generation using the `outlines` library.
/// Falls back with an error if outlines is not installed.
fn constrained_generate<'py>(
    py: Python<'py>,
    model: &PyAny,
    tokenizer: &PyAny,
    prompt: &str,
    grammar: &str,
    max_tokens: usize,
) -> PyResult<String> {
    let code = r#"
import outlines
import outlines.models as _om
import outlines.generate as _og

_mlx_model = _om.mlxlm(_model, _tokenizer)
_generator = _og.cfg(_mlx_model, _grammar)
_result = _generator(_prompt, max_tokens=_max_tokens)
"#;
    let locals = PyDict::new(py);
    locals.set_item("_model", model)?;
    locals.set_item("_tokenizer", tokenizer)?;
    locals.set_item("_grammar", grammar)?;
    locals.set_item("_prompt", prompt)?;
    locals.set_item("_max_tokens", max_tokens as i64)?;
    py.run(code, None, Some(locals))?;
    locals.get_item("_result")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no result"))?
        .extract::<String>()
}

/// Split a model response into (content, reasoning) by extracting <think>...</think> blocks.
fn extract_reasoning(response: String) -> (String, Option<String>) {
    let mut reasoning = String::new();
    let mut content = response.as_str();

    // Collect all <think>...</think> spans
    let mut remaining = content;
    let mut cleaned = String::new();
    while let Some(start) = remaining.find("<think>") {
        cleaned.push_str(&remaining[..start]);
        let after = &remaining[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            reasoning.push_str(&after[..end]);
            remaining = &after[end + "</think>".len()..];
        } else {
            // Unclosed <think> — treat rest as reasoning
            reasoning.push_str(after);
            remaining = "";
            break;
        }
    }
    cleaned.push_str(remaining);
    let cleaned = cleaned.trim().to_string();

    if reasoning.is_empty() {
        (cleaned, None)
    } else {
        (cleaned, Some(reasoning.trim().to_string()))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn count_tokens(py: Python, tokenizer: &PyAny, text: &str) -> usize {
    tokenizer
        .call_method1("encode", (text,))
        .and_then(|t| t.len())
        .unwrap_or(text.len() / 4)
}

/// Extract top-N logprobs from a full vocab logprobs array (mlx.core.array).
/// Returns a Vec of JSON values {token, logprob} sorted by descending logprob.
fn collect_top_logprobs(py: Python, tokenizer: &PyAny, logprobs_arr: PyObject, top_n: u32) -> Vec<serde_json::Value> {
    if top_n == 0 { return vec![]; }
    let result: PyResult<Vec<serde_json::Value>> = (|| {
        let mx = py.import("mlx.core")?;
        // Compute argsort(-logprobs), slice first top_n, convert to Rust vec
        let neg_lp = mx.call_method1("negative", (logprobs_arr.as_ref(py),))?;
        let sorted_idx = mx.call_method1("argsort", (neg_lp,))?;
        // Evaluate arr[:n].tolist() in Python — avoids converting all 128K entries
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("_arr", sorted_idx)?;
        locals.set_item("_n", top_n as i64)?;
        let top_idx: Vec<i64> = py.eval("_arr[:_n].tolist()", None, Some(locals))?.extract()?;

        let mut results: Vec<serde_json::Value> = Vec::with_capacity(top_n as usize);
        for &idx in &top_idx {
            let lp: f32 = logprobs_arr.as_ref(py).get_item(idx)?.extract()?;
            let tok_str: String = tokenizer
                .call_method1("convert_ids_to_tokens", (idx,))
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| format!("<{}>", idx));
            results.push(serde_json::json!({"token": tok_str, "logprob": lp}));
        }
        // Already sorted by descending logprob from argsort(-logprobs)
        Ok(results)
    })();
    result.unwrap_or_default()
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

// ── APC helpers ──────────────────────────────────────────────────────────────

fn unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Hash a slice of ChatMessages deterministically using role+content.
fn hash_messages(msgs: &[ChatMessage]) -> u64 {
    let mut h = DefaultHasher::new();
    for m in msgs {
        m.role.hash(&mut h);
        m.content.as_text().hash(&mut h);
    }
    h.finish()
}

/// Return all messages except the last user-role message.
/// This represents the "stable prefix" that is typically shared across requests
/// (e.g., the system prompt and any fixed few-shot examples).
fn extract_prefix_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let last_user = messages.iter().rposition(|m| m.role == "user");
    match last_user {
        Some(idx) if idx > 0 => messages[..idx].to_vec(),
        _ => vec![],
    }
}

/// Format a partial message list as a prompt string WITHOUT the assistant generation
/// prompt appended (i.e. add_generation_prompt=false). Used solely for token counting
/// the prefix before building an APC snapshot.
fn format_prefix_only(py: Python, tokenizer: &PyAny, messages: &[ChatMessage]) -> PyResult<String> {
    let prepared = prepare_messages(messages);
    let py_messages = PyList::new(py, prepared.iter().map(|m| {
        let d = PyDict::new(py);
        d.set_item("role", m.get("role").unwrap_or(&String::new())).unwrap();
        d.set_item("content", m.get("content").unwrap_or(&String::new())).unwrap();
        d
    }));
    let has_template = tokenizer.getattr("chat_template").map(|v| !v.is_none()).unwrap_or(false);
    if has_template && tokenizer.hasattr("apply_chat_template")? {
        let kwargs = PyDict::new(py);
        kwargs.set_item("tokenize", false)?;
        kwargs.set_item("add_generation_prompt", false)?;
        tokenizer.call_method("apply_chat_template", (py_messages,), Some(kwargs))?.extract()
    } else {
        let lines: Vec<String> = prepared.iter().map(|m| {
            format!("{}: {}", m.get("role").unwrap_or(&String::new()), m.get("content").unwrap_or(&String::new()))
        }).collect();
        Ok(lines.join("\n"))
    }
}
