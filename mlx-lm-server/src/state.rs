use crate::config::Config;
use crate::mlx_service::MlxService;
use crate::models::{BatchJob, TrainingJob};
use crate::routes::responses::{ResponsesStore, StoredResponse};
use crate::routes::vector_store::VectorStoreEntry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Count of requests currently waiting to acquire the inference semaphore.
    pub queued_requests: Arc<AtomicUsize>,
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
            queued_requests: Arc::new(AtomicUsize::new(0)),
            mlx: MlxService::new(config.clone()),
            config,
            batches: Arc::new(Mutex::new(HashMap::new())),
            vector_stores: Arc::new(Mutex::new(HashMap::new())),
            training_jobs: Arc::new(Mutex::new(HashMap::new())),
            responses: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Acquire an inference slot, tracking the queue depth for observability.
    /// Increments `queued_requests` while waiting; decrements when the permit is returned (or on error).
    pub async fn acquire_inference_slot(&self) -> Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::AcquireError> {
        self.queued_requests.fetch_add(1, Ordering::Relaxed);
        let result = self.inference_sem.clone().acquire_owned().await;
        self.queued_requests.fetch_sub(1, Ordering::Relaxed);
        result
    }
}
