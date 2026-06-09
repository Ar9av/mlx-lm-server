use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub debug: bool,
    pub cors_origins: Vec<String>,
    pub max_concurrent: usize,
    pub default_tts_model: String,
    pub default_stt_model: String,
    pub default_voice: String,
    pub default_speed: f64,
    pub default_language: String,
    pub max_tts_chars: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            host: env_str("MLX_AUDIO_HOST", "0.0.0.0"),
            port: env_u16("MLX_AUDIO_PORT", 8001),
            debug: env_bool("MLX_AUDIO_DEBUG", false),
            cors_origins: env_list("MLX_AUDIO_CORS_ORIGINS", &["http://localhost:3000", "http://localhost:5173"]),
            max_concurrent: env_usize("MLX_AUDIO_MAX_CONCURRENT", 1),
            default_tts_model: env_str("MLX_AUDIO_TTS_MODEL", "mlx-community/Kokoro-82M-bf16"),
            default_stt_model: env_str("MLX_AUDIO_STT_MODEL", "mlx-community/whisper-large-v3-turbo-asr-fp16"),
            default_voice: env_str("MLX_AUDIO_DEFAULT_VOICE", "af_heart"),
            default_speed: env_f64("MLX_AUDIO_DEFAULT_SPEED", 1.0),
            default_language: env_str("MLX_AUDIO_DEFAULT_LANGUAGE", "en"),
            max_tts_chars: env_usize("MLX_AUDIO_MAX_TTS_CHARS", 10_000),
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
