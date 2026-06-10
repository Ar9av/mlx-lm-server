use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub debug: bool,
    pub cors_origins: Vec<String>,
    pub max_concurrent: usize,
    pub default_model: String,
    pub default_steps: u32,
    pub default_guidance: f64,
    pub default_width: u32,
    pub default_height: u32,
    pub default_quantize: Option<u32>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            host: env_str("MLX_IMAGE_HOST", "0.0.0.0"),
            port: env_u16("MLX_IMAGE_PORT", 8002),
            debug: env_bool("MLX_IMAGE_DEBUG", false),
            cors_origins: env_list("MLX_IMAGE_CORS_ORIGINS", &["http://localhost:3000", "http://localhost:5173"]),
            max_concurrent: env_usize("MLX_IMAGE_MAX_CONCURRENT", 1),
            default_model: env_str("MLX_IMAGE_DEFAULT_MODEL", "flux-schnell"),
            default_steps: env_u32("MLX_IMAGE_DEFAULT_STEPS", 4),
            default_guidance: env_f64("MLX_IMAGE_DEFAULT_GUIDANCE", 4.0),
            default_width: env_u32("MLX_IMAGE_DEFAULT_WIDTH", 1024),
            default_height: env_u32("MLX_IMAGE_DEFAULT_HEIGHT", 1024),
            default_quantize: env::var("MLX_IMAGE_QUANTIZE").ok().and_then(|v| v.parse().ok()),
        }
    }
}

fn env_str(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_bool(key: &str, default: bool) -> bool {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_u16(key: &str, default: u16) -> u16 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_list(key: &str, defaults: &[&str]) -> Vec<String> {
    env::var(key)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| defaults.iter().map(|s| s.to_string()).collect())
}
