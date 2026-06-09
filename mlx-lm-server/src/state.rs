use crate::config::Config;
use crate::mlx_service::MlxService;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub mlx: MlxService,
    pub config: Arc<Config>,
    pub inference_sem: Arc<Semaphore>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let max = config.max_concurrent.max(1);
        let config = Arc::new(config);
        Self {
            inference_sem: Arc::new(Semaphore::new(max)),
            mlx: MlxService::new(config.clone()),
            config,
        }
    }
}
