use crate::config::Config;
use crate::mlx_service::MlxService;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub mlx: MlxService,
    pub config: Arc<Config>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let config = Arc::new(config);
        Self {
            mlx: MlxService::new(config.clone()),
            config,
        }
    }
}
