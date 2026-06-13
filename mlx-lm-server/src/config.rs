use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub app_name: String,
    pub debug: bool,
    pub default_model: String,
    pub max_model_size_gb: Option<f64>,
    pub allowed_models: Option<Vec<String>>,
    pub default_max_tokens: usize,
    pub default_temperature: f64,
    pub default_top_p: f64,
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub stream_timeout: f64,
    pub max_concurrent: usize,
    /// Override top_k for MoE router layers at model load time (e.g. 4 for Qwen3-MoE).
    /// Reduces router fan-out and gives ~7-16% decode speedup with minimal quality impact.
    pub moe_top_k: Option<u32>,
    /// Multiply the reported context_length in /v1/models by this factor.
    /// Use when running small-ctx models with clients that expect large context windows.
    pub context_scale: f64,
    /// Path to a JSON file containing warm-up message arrays (array of message arrays).
    /// These are prefilled at startup to prime the KV cache prefix.
    pub warm_prompts_file: Option<String>,
    /// Interval in seconds for SSE keep-alive comments during long prefill.
    pub keepalive_secs: u64,
    /// Automatically cache KV state for common prompt prefixes (system messages).
    pub enable_apc: bool,
    /// Maximum number of prefix snapshots to keep (LRU eviction when exceeded).
    pub apc_max_entries: usize,
    /// How many seconds to keep the model loaded after the last request (default 300).
    /// Requests can override per-call with the `keep_alive` field.
    /// Set to 0 to unload immediately; set to u64::MAX to never auto-unload.
    pub default_keep_alive_secs: u64,
    /// Default file path for prefix-cache save/load.  None = no default path.
    pub kv_persist_path: Option<String>,
    /// Prefill chunk size passed to generate_step (default 512).
    /// Larger = faster prefill; smaller = lower peak memory on long prompts.
    pub prefill_step_size: usize,
    /// After how many cached tokens to begin KV-cache quantization (0 = from the start).
    /// Only active when kv_bits is set (per-request or via MLX_DEFAULT_KV_BITS).
    pub quantized_kv_start: usize,
    /// Server-level default KV quantization bits (4 or 8). None = no quantization unless
    /// overridden per-request. Reduces KV memory ~2–4x with minimal quality loss.
    pub default_kv_bits: Option<u32>,
    /// Metal allocator cache limit in GB. None = server defaults to 1 GB (enough to
    /// recycle KV buffers between sequential requests without holding excess RAM).
    pub metal_cache_limit_gb: Option<f64>,
    /// Run a 1-token dummy inference immediately after every model load to pre-compile
    /// Metal shaders and warm up the KV allocator.  Eliminates the cold-start latency
    /// spike on the very first user request.  Set to false to disable.
    pub auto_warm: bool,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            app_name: env_str("MLX_APP_NAME", "MLX LM Server"),
            debug: env_bool("MLX_DEBUG", false),
            default_model: env_str(
                "MLX_DEFAULT_MODEL",
                "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            ),
            max_model_size_gb: env_opt_f64("MLX_MAX_MODEL_SIZE_GB"),
            allowed_models: env_opt_list("MLX_ALLOWED_MODELS"),
            default_max_tokens: env_usize("MLX_DEFAULT_MAX_TOKENS", 2048),
            default_temperature: env_f64("MLX_DEFAULT_TEMPERATURE", 0.7),
            default_top_p: env_f64("MLX_DEFAULT_TOP_P", 0.9),
            host: env_str("MLX_HOST", "0.0.0.0"),
            port: env_u16("MLX_PORT", 8000),
            cors_origins: env_list(
                "MLX_CORS_ORIGINS",
                &["http://localhost:5173", "http://localhost:3000"],
            ),
            stream_timeout: env_f64("MLX_STREAM_TIMEOUT", 120.0),
            max_concurrent: env_usize("MLX_MAX_CONCURRENT", 1),
            moe_top_k: env_opt_u32("MLX_MOE_TOP_K"),
            context_scale: env_f64("MLX_CONTEXT_SCALE", 1.0),
            warm_prompts_file: env_opt_str("MLX_WARM_PROMPTS"),
            keepalive_secs: env_usize("MLX_KEEPALIVE_SECS", 15) as u64,
            enable_apc: env_bool("MLX_ENABLE_APC", true),
            apc_max_entries: env_usize("MLX_APC_MAX_ENTRIES", 32),
            default_keep_alive_secs: env_usize("MLX_KEEP_ALIVE_SECS", 300) as u64,
            kv_persist_path: env_opt_str("MLX_KV_PERSIST_PATH"),
            prefill_step_size: env_usize("MLX_PREFILL_STEP_SIZE", 2048),
            quantized_kv_start: env_usize("MLX_QUANTIZED_KV_START", 0),
            default_kv_bits: env_opt_u32("MLX_DEFAULT_KV_BITS"),
            metal_cache_limit_gb: env_opt_f64("MLX_METAL_CACHE_LIMIT_GB"),
            auto_warm: env_bool("MLX_AUTO_WARM", true),
        }
    }
}

fn env_str(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u16(key: &str, default: u16) -> u16 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_opt_f64(key: &str) -> Option<f64> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_opt_u32(key: &str) -> Option<u32> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_opt_str(key: &str) -> Option<String> {
    env::var(key).ok().filter(|s| !s.is_empty())
}

fn env_opt_list(key: &str) -> Option<Vec<String>> {
    env::var(key)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
}

fn env_list(key: &str, defaults: &[&str]) -> Vec<String> {
    env::var(key)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| defaults.iter().map(|s| s.to_string()).collect())
}
