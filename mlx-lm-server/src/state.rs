use crate::config::Config;
use crate::mlx_service::MlxService;
use crate::models::{BatchJob, TrainingJob};
use crate::routes::responses::{ResponsesStore, StoredResponse};
use crate::routes::vector_store::VectorStoreEntry;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, Semaphore};

pub struct TrainingJobEntry {
    pub job: TrainingJob,
    pub tx: broadcast::Sender<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub mlx: MlxService,
    pub config: Arc<Config>,
    pub inference_sem: Arc<Semaphore>,
    pub batches: Arc<Mutex<HashMap<String, BatchJob>>>,
    pub vector_stores: Arc<Mutex<HashMap<String, VectorStoreEntry>>>,
    pub training_jobs: Arc<Mutex<HashMap<String, TrainingJobEntry>>>,
    pub responses: ResponsesStore,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let max = config.max_concurrent.max(1);
        let config = Arc::new(config);
        Self {
            inference_sem: Arc::new(Semaphore::new(max)),
            mlx: MlxService::new(config.clone()),
            config,
            batches: Arc::new(Mutex::new(HashMap::new())),
            vector_stores: Arc::new(Mutex::new(HashMap::new())),
            training_jobs: Arc::new(Mutex::new(HashMap::new())),
            responses: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
