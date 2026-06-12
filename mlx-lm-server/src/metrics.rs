use prometheus::{
    register_counter_vec, register_gauge, register_histogram_vec, CounterVec, Gauge, HistogramVec,
    TextEncoder,
};
use std::sync::OnceLock;

static REQUESTS_TOTAL: OnceLock<CounterVec> = OnceLock::new();
static REQUEST_DURATION: OnceLock<HistogramVec> = OnceLock::new();
static TOKENS_TOTAL: OnceLock<CounterVec> = OnceLock::new();
static ACTIVE_REQUESTS: OnceLock<Gauge> = OnceLock::new();

pub fn init() {
    REQUESTS_TOTAL.get_or_init(|| {
        register_counter_vec!(
            "mlx_requests_total",
            "Total HTTP requests",
            &["endpoint", "method", "status"]
        )
        .expect("failed to register mlx_requests_total")
    });
    REQUEST_DURATION.get_or_init(|| {
        register_histogram_vec!(
            "mlx_request_duration_seconds",
            "HTTP request duration in seconds",
            &["endpoint"],
            vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        .expect("failed to register mlx_request_duration_seconds")
    });
    TOKENS_TOTAL.get_or_init(|| {
        register_counter_vec!(
            "mlx_tokens_total",
            "Total tokens processed",
            &["type"]
        )
        .expect("failed to register mlx_tokens_total")
    });
    ACTIVE_REQUESTS.get_or_init(|| {
        register_gauge!("mlx_active_requests", "In-flight requests")
            .expect("failed to register mlx_active_requests")
    });
}

pub fn inc_request(endpoint: &str, method: &str, status: u16) {
    if let Some(c) = REQUESTS_TOTAL.get() {
        c.with_label_values(&[endpoint, method, &status.to_string()]).inc();
    }
}

pub fn observe_duration(endpoint: &str, secs: f64) {
    if let Some(h) = REQUEST_DURATION.get() {
        h.with_label_values(&[endpoint]).observe(secs);
    }
}

pub fn add_tokens(prompt: usize, completion: usize) {
    if let Some(c) = TOKENS_TOTAL.get() {
        c.with_label_values(&["prompt"]).inc_by(prompt as f64);
        c.with_label_values(&["completion"]).inc_by(completion as f64);
    }
}

pub fn inc_active() {
    if let Some(g) = ACTIVE_REQUESTS.get() {
        g.inc();
    }
}

pub fn dec_active() {
    if let Some(g) = ACTIVE_REQUESTS.get() {
        g.dec();
    }
}

pub fn render() -> String {
    let encoder = TextEncoder::new();
    let families = prometheus::gather();
    encoder.encode_to_string(&families).unwrap_or_default()
}
