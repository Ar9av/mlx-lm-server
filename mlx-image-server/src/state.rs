use crate::config::Config;
use crate::image_service::ImageService;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub svc: ImageService,
    pub config: Arc<Config>,
    pub semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let config = Arc::new(config);
        let max_concurrent = config.max_concurrent;
        Self {
            svc: ImageService::new(config.clone()),
            config: config.clone(),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
}
