use axum::{extract::State, http::StatusCode, response::{IntoResponse, Response}, Json};

use crate::models::{ChatMessage, MessageContent, PipelineRequest, PipelineResponse, PipelineStepResult, SamplerParams};
use crate::state::AppState;

fn apply_template(template: &str, input: &str) -> String {
    template.replace("{input}", input)
}

pub async fn run_pipeline(
    State(state): State<AppState>,
    Json(req): Json<PipelineRequest>,
) -> Response {
    if !state.mlx.is_loaded().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": {"message": "No model loaded", "type": "server_error", "code": "model_not_loaded"}})),
        ).into_response();
    }

    if req.steps.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": {"message": "steps must not be empty", "type": "invalid_request_error", "code": "invalid_request"}})),
        ).into_response();
    }

    let model_name = state.mlx.current_model().await.unwrap_or_default();
    let mut current_value = req.input.clone();
    let mut step_results: Vec<PipelineStepResult> = Vec::with_capacity(req.steps.len());
    let mut total_prompt_tokens: usize = 0;
    let mut total_completion_tokens: usize = 0;

    for step in &req.steps {
        let mut messages: Vec<ChatMessage> = Vec::new();

        if let Some(system_template) = &step.system {
            messages.push(ChatMessage {
                role: "system".into(),
                content: MessageContent::Text(apply_template(system_template, &current_value)),
            });
        }

        let user_text = match &step.user {
            Some(user_template) => apply_template(user_template, &current_value),
            None => current_value.clone(),
        };
        messages.push(ChatMessage {
            role: "user".into(),
            content: MessageContent::Text(user_text),
        });

        let max_tokens = step.max_tokens.or(req.max_tokens_per_step).unwrap_or(512);
        let temperature = step.temperature.or(req.temperature).unwrap_or(0.7);

        let sampler = SamplerParams {
            temperature,
            top_p: 0.9,
            ..Default::default()
        };

        let _permit = match state.acquire_inference_slot().await {
            Ok(p) => p,
            Err(_) => return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {"message": "inference queue closed", "type": "server_error", "code": "service_unavailable"}})),
            ).into_response(),
        };

        match state.mlx.generate_response(
            messages,
            max_tokens,
            sampler,
            serde_json::Value::Null,
            None,
            None,
            None,
            None,
            vec![],
            false,
            0,
            None,
            None,
        ).await {
            Ok((content, prompt_tokens, completion_tokens, _, _, _, _)) => {
                total_prompt_tokens += prompt_tokens;
                total_completion_tokens += completion_tokens;
                step_results.push(PipelineStepResult {
                    name: step.name.clone(),
                    output: content.clone(),
                    prompt_tokens,
                    completion_tokens,
                });
                current_value = content;
            }
            Err(e) => return e.into_response(),
        }
    }

    let output = current_value;

    (StatusCode::OK, Json(PipelineResponse {
        steps: step_results,
        output,
        model: model_name,
        total_prompt_tokens,
        total_completion_tokens,
    })).into_response()
}
