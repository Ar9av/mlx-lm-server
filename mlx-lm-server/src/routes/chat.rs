use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use tokio::time::Duration;
use tokio_stream::StreamExt as TokioStreamExt;
use tracing::{error, info};

use crate::error::MlxError;
use crate::metrics;
use crate::mlx_service::{MlxService, MAX_MESSAGE_TOKENS};
use crate::models::{ApplyTemplateRequest, ApplyTemplateResponse, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, DetokenizeRequest, DetokenizeResponse, LogprobsInfo, MessageContent, SamplerParams, StopSequence, TokenizeRequest, TokenizeResponse, TokenLogprob, TopLogprob, Tool, ToolCall, Usage};
use crate::state::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let n = req.n.unwrap_or(1).max(1).min(16);
    if req.stream.unwrap_or(false) && n > 1 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": {"message": "streaming does not support n > 1", "type": "invalid_request_error"}
        }))).into_response();
    }

    if !state.mlx.is_loaded().await {
        if let Err(e) = state.mlx.load_model(req.model.clone(), None).await {
            return e.into_response();
        }
    }

    let total_chars: usize = req.messages.iter().map(|m| m.content.as_text().len()).sum();

    // Check for image content — route to vision handler
    let all_image_urls: Vec<String> = req.messages.iter()
        .flat_map(|m| m.content.image_urls())
        .collect();
    if !all_image_urls.is_empty() {
        let max_tokens = req.max_tokens.unwrap_or(state.config.default_max_tokens);
        let temperature = req.temperature.unwrap_or(state.config.default_temperature);
        let top_p = req.top_p.unwrap_or(state.config.default_top_p);
        let _permit = match state.inference_sem.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
        };
        return match state.mlx.generate_vision_response(
            req.messages.clone(), all_image_urls, max_tokens, temperature, top_p,
        ).await {
            Ok((content, prompt_tokens, completion_tokens)) => {
                let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
                let resp = ChatCompletionResponse::new(
                    MlxService::new_chat_id(),
                    model_name,
                    content,
                    crate::models::Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
                );
                (StatusCode::OK, Json(resp)).into_response()
            }
            Err(e) => e.into_response(),
        };
    }
    if total_chars / 4 > MAX_MESSAGE_TOKENS {
        return MlxError::TokenLimit(format!(
            "estimated {} tokens exceeds maximum {}", total_chars / 4, MAX_MESSAGE_TOKENS
        )).into_response();
    }

    // Inject JSON mode system message if requested
    let mut messages = req.messages.clone();
    if let Some(rf) = &req.response_format {
        if rf.kind == "json_object" {
            let instruction = "You must respond with valid JSON only. Do not include any text before or after the JSON object.".to_string();
            if let Some(first_sys) = messages.iter_mut().find(|m| m.role == "system") {
                let existing = first_sys.content.as_text();
                first_sys.content = MessageContent::Text(format!("{}\n\n{}", existing, instruction));
            } else {
                messages.insert(0, ChatMessage { role: "system".into(), content: MessageContent::Text(instruction) });
            }
        } else if rf.kind == "json_schema" {
            let schema_hint = rf.json_schema.as_ref()
                .and_then(|s| s.get("schema"))
                .map(|s| serde_json::to_string_pretty(s).unwrap_or_default())
                .unwrap_or_default();
            let instruction = format!(
                "Respond only with valid JSON that matches this schema:\n```json\n{}\n```",
                schema_hint
            );
            if let Some(first_sys) = messages.iter_mut().find(|m| m.role == "system") {
                let existing = first_sys.content.as_text();
                first_sys.content = MessageContent::Text(format!("{}\n\n{}", existing, instruction));
            } else {
                messages.insert(0, ChatMessage { role: "system".into(), content: MessageContent::Text(instruction) });
            }
        }
    }

    let max_tokens = req.max_tokens.unwrap_or(state.config.default_max_tokens);
    let sampler = SamplerParams {
        temperature: req.temperature.unwrap_or(state.config.default_temperature),
        top_p: req.top_p.unwrap_or(state.config.default_top_p),
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        num_draft_tokens: req.num_draft_tokens,
        xtc_probability: req.xtc_probability,
        xtc_threshold: req.xtc_threshold,
        thinking_budget: req.thinking_budget.or_else(|| {
            req.reasoning_effort.as_deref().map(|e| match e {
                "low"    => 512,
                "high"   => 8192,
                _        => 2048, // "medium" and any unknown value
            })
        }),
        logit_bias: req.logit_bias.clone(),
        grammar: req.grammar.clone(),
        mirostat: req.mirostat,
        mirostat_tau: req.mirostat_tau,
        mirostat_eta: req.mirostat_eta,
        dynatemp_range: req.dynatemp_range,
        dynatemp_exponent: req.dynatemp_exponent,
    };
    let chat_template_kwargs = req.chat_template_kwargs.clone();
    let kv_bits = req.kv_bits;
    let kv_group_size = req.kv_group_size;
    let adapter_name = req.adapter_name.clone();
    let tools = req.tools.clone();
    let stop_strings = match &req.stop {
        Some(StopSequence::Single(s)) => vec![s.clone()],
        Some(StopSequence::Multiple(v)) => v.clone(),
        None => vec![],
    };
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_n_logprobs = req.top_logprobs.unwrap_or(0);
    let seed = req.seed;
    let session_id = req.session_id.clone();

    let permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "inference queue closed"}))).into_response(),
    };

    if req.stream.unwrap_or(false) {
        stream_response(state, req, messages, max_tokens, sampler, chat_template_kwargs, kv_bits, kv_group_size, adapter_name, tools, stop_strings, want_logprobs, top_n_logprobs, seed, session_id, permit).await
    } else {
        sync_response(state, req, messages, max_tokens, sampler, chat_template_kwargs, kv_bits, kv_group_size, adapter_name, tools, stop_strings, want_logprobs, top_n_logprobs, seed, session_id, permit, n).await
    }
}

async fn sync_response(
    state: AppState,
    req: ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    sampler: SamplerParams,
    chat_template_kwargs: serde_json::Value,
    kv_bits: Option<u32>,
    kv_group_size: Option<u32>,
    adapter_name: Option<String>,
    tools: Option<Vec<Tool>>,
    stop_strings: Vec<String>,
    want_logprobs: bool,
    top_n_logprobs: u32,
    seed: Option<u64>,
    session_id: Option<String>,
    _permit: tokio::sync::OwnedSemaphorePermit,
    n: u32,
) -> Response {
    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    info!("Generating {} completion(s) for model {}", n, model_name);

    let chat_id = MlxService::new_chat_id();
    let mut choices: Vec<crate::models::ChatCompletionChoice> = Vec::with_capacity(n as usize);
    let mut total_prompt = 0usize;
    let mut total_completion = 0usize;

    // Extract JSON schema for post-generation validation
    let json_validation_schema = req.response_format.as_ref().and_then(|rf| match rf.kind.as_str() {
        "json_schema" => rf.json_schema.as_ref().and_then(|s| s.get("schema")).cloned(),
        "json_object" => Some(serde_json::Value::Null), // only check parseability
        _ => None,
    });
    let validate_json = json_validation_schema.is_some();
    const MAX_JSON_RETRIES: usize = 3;

    for i in 0..n {
        let run_seed = seed.map(|s| s.wrapping_add(i as u64));
        let mut gen_result = None;

        for attempt in 0..if validate_json { MAX_JSON_RETRIES } else { 1 } {
            let result = state.mlx.generate_response(
                messages.clone(), max_tokens, sampler.clone(), chat_template_kwargs.clone(),
                kv_bits, kv_group_size, adapter_name.clone(),
                tools.clone(), stop_strings.clone(),
                want_logprobs, top_n_logprobs,
                run_seed,
                if i == 0 { session_id.clone() } else { None },
            ).await;

            match result {
                Err(e) => {
                    error!("Generation run {} attempt {} failed: {}", i, attempt, e);
                    return e.into_response();
                }
                Ok(tuple) => {
                    let json_ok = if validate_json && tuple.3.is_empty() {
                        match state.mlx.validate_json_schema(
                            tuple.0.clone(),
                            json_validation_schema.clone(),
                        ).await {
                            Ok(()) => true,
                            Err(err) if attempt + 1 < MAX_JSON_RETRIES => {
                                info!("JSON validation failed attempt {} (retrying): {}", attempt, err);
                                false
                            }
                            Err(err) => {
                                info!("JSON validation failed on final attempt, using output anyway: {}", err);
                                true
                            }
                        }
                    } else {
                        true
                    };

                    if json_ok {
                        gen_result = Some(tuple);
                        break;
                    }
                }
            }
        }

        let (mut content, prompt_tokens, completion_tokens, tool_calls, finish_reason, lp_list, reasoning) =
            gen_result.expect("generation loop must yield a result");

        // For JSON output modes, strip model artifacts (code fences, EOS tokens)
        if validate_json && tool_calls.is_empty() {
            let s = content
                .replace("<|im_end|>", "")
                .replace("<|endoftext|>", "")
                .replace("<|end|>", "")
                .replace("</s>", "");
            let trimmed = s.trim();
            let inner = if let Some(r) = trimmed.strip_prefix("```json") {
                r.trim_end_matches('`').trim()
            } else if let Some(r) = trimmed.strip_prefix("```") {
                r.trim_end_matches('`').trim()
            } else {
                trimmed
            };
            content = inner.to_string();
        }

        metrics::add_tokens(prompt_tokens, completion_tokens);
        total_prompt += prompt_tokens;
        total_completion += completion_tokens;

        let message = if !tool_calls.is_empty() {
            crate::models::ChatMessage { role: "assistant".into(), content: crate::models::MessageContent::Text(String::new()) }
        } else {
            crate::models::ChatMessage { role: "assistant".into(), content: crate::models::MessageContent::Text(content) }
        };

        let logprobs = lp_list.map(|lps| LogprobsInfo {
            content: lps.into_iter().map(|v| TokenLogprob {
                token: v["token"].as_str().unwrap_or("").to_string(),
                logprob: v["logprob"].as_f64().unwrap_or(0.0) as f32,
                bytes: None,
                top_logprobs: v["top_logprobs"].as_array().unwrap_or(&vec![]).iter().map(|t| TopLogprob {
                    token: t["token"].as_str().unwrap_or("").to_string(),
                    logprob: t["logprob"].as_f64().unwrap_or(0.0) as f32,
                    bytes: None,
                }).collect(),
            }).collect(),
        });

        choices.push(crate::models::ChatCompletionChoice {
            index: i,
            message,
            finish_reason: Some(finish_reason),
            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
            logprobs,
            reasoning_content: reasoning,
        });
    }

    let usage = Usage { prompt_tokens: total_prompt, completion_tokens: total_completion, total_tokens: total_prompt + total_completion };
    let resp = crate::models::ChatCompletionResponse {
        id: chat_id,
        object: "chat.completion",
        created: crate::models::now_secs_pub(),
        model: model_name,
        choices,
        usage,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

const TOOL_SENTINEL: &str = "\x00TOOL_CALLS:";
const FINISH_SENTINEL: &str = "\x00FINISH:";
const LP_SENTINEL: &str = "\x00LP:";

async fn stream_response(
    state: AppState,
    req: ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    sampler: SamplerParams,
    chat_template_kwargs: serde_json::Value,
    kv_bits: Option<u32>,
    kv_group_size: Option<u32>,
    adapter_name: Option<String>,
    tools: Option<Vec<Tool>>,
    stop_strings: Vec<String>,
    want_logprobs: bool,
    top_n_logprobs: u32,
    seed: Option<u64>,
    session_id: Option<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let model_name = state.mlx.current_model().await.unwrap_or(req.model.clone());
    let chat_id = MlxService::new_chat_id();
    let timeout = state.config.stream_timeout;

    let (token_stream, session_info) = match state.mlx.generate_stream(
        messages, max_tokens, sampler, timeout, chat_template_kwargs, kv_bits, kv_group_size, adapter_name, tools, stop_strings, want_logprobs, top_n_logprobs, seed, session_id,
    ).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };

    let chat_id_clone = chat_id.clone();
    let model_clone = model_name.clone();
    let mut first = true;
    // Shared finish_reason updated when sentinel arrives; used in final stop chunk
    let finish_reason = std::sync::Arc::new(std::sync::Mutex::new("stop".to_string()));
    let finish_reason_w = finish_reason.clone();
    // Pending logprob for the next text token
    let mut pending_lp: Option<LogprobsInfo> = None;
    // Reasoning state machine: track whether we're inside <think>...</think>
    let mut in_think = false;
    let mut think_buf = String::new(); // partial tag accumulation for split-chunk detection

    // Wrap the token stream with keep-alive timeouts so the client doesn't disconnect
    // during long prefill phases. SSE comment lines are ignored by all clients.
    let keepalive_dur = Duration::from_secs(state.config.keepalive_secs);
    let token_stream_timed = TokioStreamExt::timeout(token_stream, keepalive_dur);

    let sse_stream = futures::StreamExt::map(token_stream_timed, move |timed| -> Result<Bytes, std::io::Error> {
        // Timeout = no token for keepalive_dur; emit SSE comment
        let result = match timed {
            Err(_elapsed) => return Ok(Bytes::from(": keep-alive\n\n")),
            Ok(r) => r,
        };
        let data = match result {
            Ok(token) => {
                if token.starts_with(FINISH_SENTINEL) {
                    let reason = token
                        .trim_start_matches(FINISH_SENTINEL)
                        .trim_end_matches('\x00')
                        .to_string();
                    if let Ok(mut guard) = finish_reason_w.lock() {
                        *guard = reason;
                    }
                    return Ok(Bytes::new());
                }
                if token.starts_with(LP_SENTINEL) {
                    let json_part = token.trim_start_matches(LP_SENTINEL).trim_end_matches('\x00');
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_part) {
                        let tok = v["token"].as_str().unwrap_or("").to_string();
                        let lp = v["logprob"].as_f64().unwrap_or(0.0) as f32;
                        let top = v["top_logprobs"].as_array().unwrap_or(&vec![]).iter().map(|t| TopLogprob {
                            token: t["token"].as_str().unwrap_or("").to_string(),
                            logprob: t["logprob"].as_f64().unwrap_or(0.0) as f32,
                            bytes: None,
                        }).collect();
                        pending_lp = Some(LogprobsInfo { content: vec![TokenLogprob { token: tok, logprob: lp, bytes: None, top_logprobs: top }] });
                    }
                    return Ok(Bytes::new());
                }
                if token.starts_with(TOOL_SENTINEL) {
                    let json_part = token
                        .trim_start_matches(TOOL_SENTINEL)
                        .trim_end_matches('\x00');
                    let tool_calls: Vec<ToolCall> = serde_json::from_str(json_part).unwrap_or_default();
                    let chunk = ChatCompletionChunk::tool_calls_chunk(&chat_id_clone, &model_clone, tool_calls);
                    serde_json::to_string(&chunk).unwrap_or_default()
                } else {
                    // Reasoning state machine: route <think> content to reasoning_content delta
                    let (content_text, reasoning_text, new_in_think) =
                        classify_token(&token, &mut think_buf, in_think);
                    in_think = new_in_think;

                    let lp = pending_lp.take();
                    let is_first = first;
                    if is_first && (content_text.is_some() || reasoning_text.is_some()) {
                        first = false;
                    }
                    let chunk = if let Some(rt) = reasoning_text {
                        ChatCompletionChunk::token_lp_reasoning(
                            &chat_id_clone, &model_clone, &rt, is_first, lp, true,
                        )
                    } else {
                        let ct = content_text.unwrap_or_default();
                        ChatCompletionChunk::token_lp(
                            &chat_id_clone, &model_clone, &ct, is_first, lp,
                        )
                    };
                    serde_json::to_string(&chunk).unwrap_or_default()
                }
            }
            Err(e) => serde_json::json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            }).to_string(),
        };
        Ok(Bytes::from(format!("data: {}\n\n", data)))
    });

    let mlx_svc = state.mlx.clone();
    let combined = futures::StreamExt::chain(sse_stream, futures::stream::once(async move {
        let reason = finish_reason.lock().map(|g| g.clone()).unwrap_or_else(|_| "stop".into());
        let final_chunk = ChatCompletionChunk::finish(&chat_id, &model_name, &reason);
        let final_data = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&final_chunk).unwrap_or_default()
        );
        drop(permit);
        // Commit session KV cache after stream ends (Python mutated it in-place)
        if let Some((sid, kv_cache, _)) = session_info {
            mlx_svc.commit_session(sid, kv_cache).await;
        }
        Ok::<Bytes, std::io::Error>(Bytes::from(final_data))
    }));

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));
    headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));

    (headers, axum::body::Body::from_stream(combined)).into_response()
}

pub async fn tokenize(
    State(state): State<AppState>,
    Json(req): Json<TokenizeRequest>,
) -> impl IntoResponse {
    match state.mlx.tokenize(req.prompt).await {
        Ok(tokens) => {
            let count = tokens.len();
            (StatusCode::OK, Json(TokenizeResponse { tokens, count })).into_response()
        }
        Err(e) => e.into_response(),
    }
}

pub async fn delete_session(
    State(state): State<AppState>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    state.mlx.delete_prompt_session(&session_id).await;
    (StatusCode::OK, Json(serde_json::json!({"deleted": session_id}))).into_response()
}

pub async fn detokenize(
    State(state): State<AppState>,
    Json(req): Json<DetokenizeRequest>,
) -> impl IntoResponse {
    match state.mlx.detokenize(req.tokens).await {
        Ok(text) => (StatusCode::OK, Json(DetokenizeResponse { text })).into_response(),
        Err(e) => e.into_response(),
    }
}

pub async fn apply_template(
    State(state): State<AppState>,
    Json(req): Json<ApplyTemplateRequest>,
) -> impl IntoResponse {
    match state.mlx.apply_chat_template_prompt(req.messages, req.add_generation_prompt).await {
        Ok((prompt, token_count)) => {
            (StatusCode::OK, Json(ApplyTemplateResponse { prompt, token_count })).into_response()
        }
        Err(e) => e.into_response(),
    }
}

/// Route a streaming token to either content or reasoning_content based on <think> tag state.
/// Returns (content, reasoning, new_in_think). The tag accumulation buffer handles tokens
/// that arrive in the middle of a `<think>` or `</think>` tag split across chunks.
fn classify_token(
    token: &str,
    buf: &mut String,
    in_think: bool,
) -> (Option<String>, Option<String>, bool) {
    buf.push_str(token);

    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut content_out = String::new();
    let mut reasoning_out = String::new();
    let mut state = in_think;
    let mut remaining = buf.as_str();

    loop {
        if state {
            // Inside reasoning — look for </think>
            if let Some(pos) = remaining.find(CLOSE) {
                reasoning_out.push_str(&remaining[..pos]);
                remaining = &remaining[pos + CLOSE.len()..];
                state = false;
            } else if remaining.ends_with_partial_prefix(CLOSE) {
                // Partial </think> at end — keep in buffer, don't emit yet
                let safe_end = remaining.len().saturating_sub(CLOSE.len() - 1);
                reasoning_out.push_str(&remaining[..safe_end]);
                remaining = &remaining[safe_end..];
                break;
            } else {
                reasoning_out.push_str(remaining);
                remaining = "";
                break;
            }
        } else {
            // Outside reasoning — look for <think>
            if let Some(pos) = remaining.find(OPEN) {
                content_out.push_str(&remaining[..pos]);
                remaining = &remaining[pos + OPEN.len()..];
                state = true;
            } else if remaining.ends_with_partial_prefix(OPEN) {
                let safe_end = remaining.len().saturating_sub(OPEN.len() - 1);
                content_out.push_str(&remaining[..safe_end]);
                remaining = &remaining[safe_end..];
                break;
            } else {
                content_out.push_str(remaining);
                remaining = "";
                break;
            }
        }
    }

    *buf = remaining.to_string();

    let content = if content_out.is_empty() { None } else { Some(content_out) };
    let reasoning = if reasoning_out.is_empty() { None } else { Some(reasoning_out) };
    (content, reasoning, state)
}

trait EndsWithPartialPrefix {
    fn ends_with_partial_prefix(&self, prefix: &str) -> bool;
}

impl EndsWithPartialPrefix for str {
    fn ends_with_partial_prefix(&self, prefix: &str) -> bool {
        // Check if self ends with any non-empty prefix of `prefix`
        let bytes = self.as_bytes();
        let prefix_bytes = prefix.as_bytes();
        for len in (1..prefix_bytes.len()).rev() {
            if bytes.ends_with(&prefix_bytes[..len]) {
                return true;
            }
        }
        false
    }
}
