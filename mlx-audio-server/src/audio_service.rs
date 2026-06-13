use crate::config::Config;
use crate::error::AudioError;
use crate::models::TranscriptionSegment;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, error};

struct LoadedTts {
    model: Py<PyAny>,
    model_id: String,
    sample_rate: u32,
    loaded_at_unix: u64,
}

struct LoadedStt {
    model: Py<PyAny>,
    model_id: String,
    loaded_at_unix: u64,
}

struct LoadedSts {
    model: Py<PyAny>,
    processor: Py<PyAny>,
    model_id: String,
    loaded_at_unix: u64,
}

struct LoadedVad {
    model: Py<PyAny>,
    model_id: String,
    loaded_at_unix: u64,
}

struct Inner {
    tts: Option<LoadedTts>,
    stt: Option<LoadedStt>,
    sts: Option<LoadedSts>,
    vad: Option<LoadedVad>,
}

#[derive(Clone)]
pub struct AudioService {
    inner: Arc<Mutex<Inner>>,
    config: Arc<Config>,
}

impl AudioService {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { tts: None, stt: None, sts: None, vad: None })),
            config,
        }
    }

    // ── Load ──────────────────────────────────────────────────────────────────

    pub async fn load_tts(&self, model_id: String) -> Result<(), AudioError> {
        info!("Loading TTS model: {}", model_id);
        let mid = model_id.clone();
        let (model_py, sample_rate) = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(PyObject, u32)> {
                let mlx_audio_tts = py.import("mlx_audio.tts")?;
                let model = mlx_audio_tts.getattr("load")?.call1((&mid,))?;
                let sr: u32 = model.getattr("sample_rate")
                    .and_then(|v| v.extract())
                    .unwrap_or(24000);
                Ok((model.into(), sr))
            })
        })
        .await
        .map_err(|e| AudioError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| AudioError::LoadFailed(e.to_string()))?;

        let mut guard = self.inner.lock().await;
        guard.tts = Some(LoadedTts {
            model: model_py,
            model_id,
            sample_rate,
            loaded_at_unix: unix_now(),
        });
        info!("TTS model loaded");
        Ok(())
    }

    pub async fn load_stt(&self, model_id: String) -> Result<(), AudioError> {
        info!("Loading STT model: {}", model_id);
        let mid = model_id.clone();
        let model_py = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<PyObject> {
                let mlx_audio_stt = py.import("mlx_audio.stt")?;
                let model = mlx_audio_stt.getattr("load")?.call1((&mid,))?;
                Ok(model.into())
            })
        })
        .await
        .map_err(|e| AudioError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| AudioError::LoadFailed(e.to_string()))?;

        let mut guard = self.inner.lock().await;
        guard.stt = Some(LoadedStt { model: model_py, model_id, loaded_at_unix: unix_now() });
        info!("STT model loaded");
        Ok(())
    }

    pub async fn load_sts(&self, model_id: String) -> Result<(), AudioError> {
        info!("Loading STS model: {}", model_id);
        let mid = model_id.clone();
        let (model_py, processor_py) = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(PyObject, PyObject)> {
                let sts = py.import("mlx_audio.sts")?;
                // SAMAudio accepts file-path lists directly in separate_long,
                // but we also load the processor for numpy-array inputs.
                let model = sts.getattr("SAMAudio")?
                    .call_method1("from_pretrained", (&mid,))?;
                let processor = sts.getattr("SAMAudioProcessor")?
                    .call_method1("from_pretrained", (&mid,))?;
                Ok((model.into(), processor.into()))
            })
        })
        .await
        .map_err(|e| AudioError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| AudioError::LoadFailed(e.to_string()))?;

        let mut guard = self.inner.lock().await;
        guard.sts = Some(LoadedSts {
            model: model_py,
            processor: processor_py,
            model_id,
            loaded_at_unix: unix_now(),
        });
        info!("STS model loaded");
        Ok(())
    }

    // ── Info ──────────────────────────────────────────────────────────────────

    pub async fn tts_model_id(&self) -> Option<String> {
        self.inner.lock().await.tts.as_ref().map(|m| m.model_id.clone())
    }

    pub async fn stt_model_id(&self) -> Option<String> {
        self.inner.lock().await.stt.as_ref().map(|m| m.model_id.clone())
    }

    pub async fn sts_model_id(&self) -> Option<String> {
        self.inner.lock().await.sts.as_ref().map(|m| m.model_id.clone())
    }

    pub async fn unload_tts(&self) { self.inner.lock().await.tts = None; }
    pub async fn unload_stt(&self) { self.inner.lock().await.stt = None; }
    pub async fn unload_sts(&self) { self.inner.lock().await.sts = None; }
    pub async fn unload_vad(&self) { self.inner.lock().await.vad = None; }
    pub async fn vad_model_id(&self) -> Option<String> {
        self.inner.lock().await.vad.as_ref().map(|v| v.model_id.clone())
    }

    pub async fn load_vad(&self, model_id: String) -> Result<(), AudioError> {
        info!("Loading VAD model: {}", model_id);
        let mid = model_id.clone();
        let model_py = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<PyObject> {
                let vad_mod = py.import("mlx_audio.vad")?;
                let model = vad_mod.getattr("load")?.call1((&mid,))?;
                Ok(model.into())
            })
        })
        .await
        .map_err(|e| AudioError::LoadFailed(e.to_string()))?
        .map_err(|e: PyErr| AudioError::LoadFailed(e.to_string()))?;

        self.inner.lock().await.vad = Some(LoadedVad { model: model_py, model_id, loaded_at_unix: unix_now() });
        info!("VAD model loaded");
        Ok(())
    }

    /// Detect speech segments in an audio file.
    /// Returns a list of (start_secs, end_secs) tuples and the total audio duration in seconds.
    pub async fn detect_speech_segments(
        &self,
        audio_path: String,
        threshold: Option<f64>,
        min_speech_ms: Option<u64>,
        min_silence_ms: Option<u64>,
        speech_pad_ms: Option<u64>,
        return_seconds: bool,
    ) -> Result<(Vec<(f64, f64)>, f64), AudioError> {
        let guard = self.inner.lock().await;
        let vad = guard.vad.as_ref().ok_or_else(|| AudioError::Internal("VAD model not loaded".into()))?;
        let model_py = Python::with_gil(|py| vad.model.clone_ref(py));
        drop(guard);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(Vec<(f64, f64)>, f64)> {
                let model = model_py.as_ref(py);
                let kwargs = PyDict::new(py);
                if let Some(t) = threshold { kwargs.set_item("threshold", t)?; }
                if let Some(ms) = min_speech_ms { kwargs.set_item("min_speech_duration_ms", ms)?; }
                if let Some(ms) = min_silence_ms { kwargs.set_item("min_silence_duration_ms", ms)?; }
                if let Some(ms) = speech_pad_ms { kwargs.set_item("speech_pad_ms", ms)?; }
                kwargs.set_item("return_seconds", return_seconds)?;

                let output = model.call_method("generate", (&audio_path,), Some(kwargs))?;

                // Read timestamps list from VADOutput.timestamps
                let ts_list = output.getattr("timestamps")?;
                let length = ts_list.len().unwrap_or(0);
                let mut segments: Vec<(f64, f64)> = Vec::new();
                for i in 0..length {
                    let item: &pyo3::PyAny = ts_list.get_item(i)?;
                    let start: f64 = item.get_item("start")
                        .or_else(|_| item.getattr("start"))
                        .and_then(|v: &pyo3::PyAny| v.extract::<f64>())?;
                    let end: f64 = item.get_item("end")
                        .or_else(|_| item.getattr("end"))
                        .and_then(|v: &pyo3::PyAny| v.extract::<f64>())?;
                    segments.push((start, end));
                }

                // Compute duration via soundfile or fall back to last segment end
                let duration = py.import("soundfile")
                    .and_then(|sf| {
                        let info = sf.getattr("info")?.call1((&audio_path,))?;
                        info.getattr("duration")?.extract::<f64>()
                    })
                    .unwrap_or_else(|_| {
                        segments.last().map(|(_, e)| *e).unwrap_or(0.0)
                    });

                Ok((segments, duration))
            })
        })
        .await
        .map_err(|e| AudioError::Internal(e.to_string()))?
        .map_err(|e: PyErr| AudioError::Internal(e.to_string()))
    }

    // ── TTS synthesis ─────────────────────────────────────────────────────────

    pub async fn synthesize(
        &self,
        text: String,
        voice: String,
        speed: f64,
        language: Option<String>,
        ref_audio: Option<String>,
        ref_text: Option<String>,
        response_format: String,
    ) -> Result<Vec<u8>, AudioError> {
        if text.len() > self.config.max_tts_chars {
            return Err(AudioError::InputTooLong(text.len(), self.config.max_tts_chars));
        }

        let guard = self.inner.lock().await;
        let tts = guard.tts.as_ref().ok_or(AudioError::TtsNotLoaded)?;
        let model_ref = Python::with_gil(|py| tts.model.clone_ref(py));
        let sample_rate = tts.sample_rate;
        drop(guard);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<u8>> {
                let mx = py.import("mlx.core")?;
                mx.call_method1("eval", (mx.call_method1("zeros", (1_i32,))?,))?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("voice", voice.as_str())?;
                kwargs.set_item("speed", speed)?;
                // Derive lang_code from voice prefix (af_/am_ → "a", bf_/bm_ → "b")
                // or use explicit language override. Fallback "a" avoids espeak dependency.
                let lang_code = language.as_deref()
                    .unwrap_or_else(|| lang_code_from_voice(&voice));
                kwargs.set_item("lang_code", lang_code)?;
                if let Some(ref ra) = ref_audio {
                    kwargs.set_item("ref_audio", ra.as_str())?;
                }
                if let Some(ref rt) = ref_text {
                    kwargs.set_item("ref_text", rt.as_str())?;
                }
                kwargs.set_item("verbose", false)?;

                // Collect all audio chunks
                let generator = model_ref.as_ref(py)
                    .call_method("generate", (&text,), Some(kwargs))?;

                let np = py.import("numpy")?;
                let chunks = pyo3::types::PyList::empty(py);
                let mut count = 0usize;

                for item in generator.iter()? {
                    let result = item?;
                    let chunk = result.getattr("audio")?;
                    chunks.append(chunk)?;
                    count += 1;
                }

                if count == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("No audio generated"));
                }

                // Concatenate chunks
                let audio = if count == 1 {
                    chunks.get_item(0)?.into_py(py)
                } else {
                    np.call_method1("concatenate", (chunks,))?.into_py(py)
                };

                // Convert to bytes based on format
                audio_to_bytes(py, audio.as_ref(py), sample_rate, &response_format)
            })
        })
        .await
        .map_err(|e| AudioError::AudioProcessing(e.to_string()))?
        .map_err(|e: PyErr| AudioError::AudioProcessing(e.to_string()))
    }

    pub async fn synthesize_stream(
        &self,
        text: String,
        voice: String,
        speed: f64,
        language: Option<String>,
    ) -> Result<ReceiverStream<Result<Vec<u8>, AudioError>>, AudioError> {
        if text.len() > self.config.max_tts_chars {
            return Err(AudioError::InputTooLong(text.len(), self.config.max_tts_chars));
        }

        let guard = self.inner.lock().await;
        let tts = guard.tts.as_ref().ok_or(AudioError::TtsNotLoaded)?;
        let model_ref = Python::with_gil(|py| tts.model.clone_ref(py));
        let sample_rate = tts.sample_rate;
        drop(guard);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, AudioError>>(16);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let run = || -> PyResult<()> {
                    let mx = py.import("mlx.core")?;
                    mx.call_method1("eval", (mx.call_method1("zeros", (1_i32,))?,))?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("voice", voice.as_str())?;
                    kwargs.set_item("speed", speed)?;
                    let lang_code = language.as_deref()
                        .unwrap_or_else(|| lang_code_from_voice(&voice));
                    kwargs.set_item("lang_code", lang_code)?;
                    kwargs.set_item("verbose", false)?;
                    kwargs.set_item("stream", true)?;
                    kwargs.set_item("streaming_interval", 1.0f64)?;

                    let generator = model_ref.as_ref(py)
                        .call_method("generate", (&text,), Some(kwargs))?;

                    let mut first_chunk = true;
                    for item in generator.iter()? {
                        let result = item?;
                        let chunk = result.getattr("audio")?;
                        let bytes = chunk_to_wav_bytes(py, chunk, sample_rate, first_chunk)?;
                        first_chunk = false;
                        if tx.blocking_send(Ok(bytes)).is_err() {
                            break;
                        }
                    }
                    Ok(())
                };

                if let Err(e) = run() {
                    error!("TTS stream error: {}", e);
                    let _ = tx.blocking_send(Err(AudioError::AudioProcessing(e.to_string())));
                }
            });
        });

        Ok(ReceiverStream::new(rx))
    }

    // ── STT transcription ─────────────────────────────────────────────────────

    pub async fn transcribe(
        &self,
        audio_path: String,
        language: Option<String>,
        prompt: Option<String>,
        temperature: f64,
        with_segments: bool,
    ) -> Result<(String, Vec<TranscriptionSegment>, Option<String>), AudioError> {
        let guard = self.inner.lock().await;
        let stt = guard.stt.as_ref().ok_or(AudioError::SttNotLoaded)?;
        let model_ref = Python::with_gil(|py| stt.model.clone_ref(py));
        drop(guard);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<(String, Vec<TranscriptionSegment>, Option<String>)> {
                let mx = py.import("mlx.core")?;
                mx.call_method1("eval", (mx.call_method1("zeros", (1_i32,))?,))?;
                let kwargs = PyDict::new(py);
                if let Some(ref lang) = language {
                    kwargs.set_item("language", lang.as_str())?;
                }
                if let Some(ref ctx) = prompt {
                    kwargs.set_item("context", ctx.as_str())?;
                }
                kwargs.set_item("temperature", temperature)?;
                kwargs.set_item("verbose", false)?;

                let result = model_ref.as_ref(py)
                    .call_method("generate", (&audio_path,), Some(kwargs))?;

                let text: String = result.getattr("text")?.extract()?;

                let lang_out: Option<String> = result.getattr("language")
                    .ok()
                    .and_then(|v| v.extract().ok());

                let segments = if with_segments {
                    extract_segments(py, result)?
                } else {
                    vec![]
                };

                Ok((text, segments, lang_out))
            })
        })
        .await
        .map_err(|e| AudioError::Transcription(e.to_string()))?
        .map_err(|e: PyErr| AudioError::Transcription(e.to_string()))
    }

    // ── STS source separation ─────────────────────────────────────────────────

    pub async fn separate(
        &self,
        audio_path: String,
        description: String,
        output_target: String,
        output_residual: String,
    ) -> Result<(), AudioError> {
        let guard = self.inner.lock().await;
        let sts = guard.sts.as_ref().ok_or(AudioError::StsNotLoaded)?;
        let model_ref = Python::with_gil(|py| sts.model.clone_ref(py));
        let processor_ref = Python::with_gil(|py| sts.processor.clone_ref(py));
        drop(guard);

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<()> {
                // MLX Metal streams are thread-local; force init on this thread
                // before touching any model arrays from another thread.
                let mx = py.import("mlx.core")?;
                mx.call_method1("eval", (mx.call_method1("zeros", (1_i32,))?,))?;

                let sts_mod = py.import("mlx_audio.sts")?;

                // separate_long accepts List[str] paths directly
                let sep_kwargs = PyDict::new(py);
                sep_kwargs.set_item("descriptions", vec![description.as_str()])?;
                sep_kwargs.set_item("chunk_seconds", 10.0f64)?;
                sep_kwargs.set_item("verbose", false)?;

                let audios = pyo3::types::PyList::new(py, [audio_path.as_str()]);
                let result = model_ref.as_ref(py)
                    .call_method("separate_long", (audios,), Some(sep_kwargs))?;

                let save_audio = sts_mod.getattr("save_audio")?;
                let target = result.getattr("target")?.get_item(0)?;
                let residual = result.getattr("residual")?.get_item(0)?;

                // save_audio(array, path, sample_rate=48000)
                let sr = model_ref.as_ref(py).getattr("sample_rate")
                    .and_then(|v| v.extract::<u32>())
                    .unwrap_or(48000);
                save_audio.call1((target, output_target.as_str(), sr))?;
                save_audio.call1((residual, output_residual.as_str(), sr))?;

                Ok(())
            })
        })
        .await
        .map_err(|e| AudioError::AudioProcessing(e.to_string()))?
        .map_err(|e: PyErr| AudioError::AudioProcessing(e.to_string()))
    }
}

// ── Audio helpers ─────────────────────────────────────────────────────────────

// Infer Kokoro lang_code from voice prefix so callers don't need to set it.
// af_/am_ → American English "a", bf_/bm_ → British English "b", rest → "a".
fn lang_code_from_voice(voice: &str) -> &'static str {
    match voice.get(..2) {
        Some("af") | Some("am") => "a",
        Some("bf") | Some("bm") => "b",
        Some("jf") | Some("jm") => "j",
        Some("zf") | Some("zm") => "z",
        _ => "a",
    }
}

fn audio_to_bytes(
    py: Python<'_>,
    audio: &PyAny,
    sample_rate: u32,
    format: &str,
) -> PyResult<Vec<u8>> {
    let np = py.import("numpy")?;
    let audio_f32 = np.call_method1("asarray", (audio,))?
        .call_method1("astype", (np.getattr("float32")?,))?;

    match format {
        "wav" | "wave" => {
            // Scale to int16 via numpy
            let scaled = np.call_method1("multiply", (audio_f32, 32767.0_f64))?;
            let audio_i16 = scaled.call_method("clip", (-32768_i32, 32767_i32), None)?
                .call_method1("astype", (np.getattr("int16")?,))?;
            let samples: Vec<i16> = audio_i16.extract()?;

            // Build WAV in Rust (no Python wave module needed)
            let data_len = (samples.len() * 2) as u32;
            let chunk_size = data_len + 36;
            let mut out = Vec::with_capacity(44 + samples.len() * 2);
            out.extend_from_slice(b"RIFF");
            out.extend_from_slice(&chunk_size.to_le_bytes());
            out.extend_from_slice(b"WAVE");
            out.extend_from_slice(b"fmt ");
            out.extend_from_slice(&16u32.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());  // PCM
            out.extend_from_slice(&1u16.to_le_bytes());  // mono
            out.extend_from_slice(&sample_rate.to_le_bytes());
            out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
            out.extend_from_slice(&2u16.to_le_bytes());  // block align
            out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
            out.extend_from_slice(b"data");
            out.extend_from_slice(&data_len.to_le_bytes());
            for s in samples {
                out.extend_from_slice(&s.to_le_bytes());
            }
            Ok(out)
        }
        "mp3" | "flac" | "ogg" | "opus" => {
            let io_mod = py.import("io")?;
            let sf = py.import("soundfile").map_err(|_| {
                pyo3::exceptions::PyImportError::new_err(
                    format!("{} format requires soundfile: pip install soundfile", format)
                )
            })?;
            let buf = io_mod.call_method0("BytesIO")?;
            let audio_np = np.call_method1("asarray", (audio,))?;
            let sf_kwargs = PyDict::new(py);
            sf_kwargs.set_item("format", format.to_uppercase().as_str())?;
            sf.call_method("write", (buf, audio_np, sample_rate), Some(sf_kwargs))?;
            buf.call_method1("seek", (0_i32,))?;
            let result: Vec<u8> = buf.call_method0("read")?.extract()?;
            Ok(result)
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported format: {}", other))),
    }
}

fn chunk_to_wav_bytes(
    py: Python<'_>,
    audio: &PyAny,
    sample_rate: u32,
    include_header: bool,
) -> PyResult<Vec<u8>> {
    let np = py.import("numpy")?;
    let audio_f32 = np.call_method1("asarray", (audio,))?
        .call_method1("astype", (np.getattr("float32")?,))?;
    let clipped = audio_f32.call_method("clip", (-1.0_f64, 1.0_f64), None)?;
    let scaled = np.call_method1("multiply", (clipped, 32767.0_f64))?;
    let audio_i16_arr = scaled.call_method1("astype", (np.getattr("int16")?,))?;
    let raw: Vec<i16> = audio_i16_arr.extract()?;

    let mut out = Vec::new();
    if include_header {
        // Use 0xFFFFFFFF sentinel for streaming (size unknown at header time)
        let data_len = u32::MAX;
        let chunk_size = u32::MAX; // also sentinel — client ignores for streaming
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&chunk_size.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());  // PCM
        out.extend_from_slice(&1u16.to_le_bytes());  // mono
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        out.extend_from_slice(&2u16.to_le_bytes());  // block align
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
    }
    for s in raw {
        out.extend_from_slice(&(s as i16).to_le_bytes());
    }
    Ok(out)
}

fn extract_segments(py: Python<'_>, result: &PyAny) -> PyResult<Vec<TranscriptionSegment>> {
    let segs_obj = match result.getattr("segments") {
        Ok(s) if !s.is_none() => s,
        _ => return Ok(vec![]),
    };
    let segs: Vec<&PyAny> = segs_obj.extract().unwrap_or_default();
    let mut out = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        let start = seg.get_item("start")
            .or_else(|_| seg.getattr("start"))
            .and_then(|v| v.extract::<f64>())
            .unwrap_or(0.0);
        let end = seg.get_item("end")
            .or_else(|_| seg.getattr("end"))
            .and_then(|v| v.extract::<f64>())
            .unwrap_or(0.0);
        let text = seg.get_item("text")
            .or_else(|_| seg.getattr("text"))
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        out.push(TranscriptionSegment { id: i as u32, start, end, text });
    }
    Ok(out)
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub fn process_rss_mb() -> f64 {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|rss| rss as f64 / 1024.0)
        .unwrap_or(0.0)
}
