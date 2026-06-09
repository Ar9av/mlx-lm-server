use crate::audio_service::AudioService;
use crate::config::Config;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub audio: AudioService,
    pub config: Arc<Config>,
    pub semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let config = Arc::new(config);
        let max_concurrent = config.max_concurrent;
        Self {
            audio: AudioService::new(config.clone()),
            config: config.clone(),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
}
