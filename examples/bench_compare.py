#!/usr/bin/env python3
"""
Comprehensive benchmark: mlx-lm-server vs raw mlx_lm baseline.

Measures separately:
  - Model load time
  - Prefill throughput (prompt tokens / TTFT)
  - Decode throughput (completion tokens / decode time)
  - p50 / p95 latency over N runs
  - Concurrent throughput

Also tests optional optimizations:
  - KV quantization (kv_bits=4, kv_bits=8)
  - Larger prefill_step_size (server env var MLX_PREFILL_STEP_SIZE)

Usage:
  # Start server first (or let this script start it):
  ./run.sh lm

  # Run benchmark against a running server:
  python examples/bench_compare.py --server-only

  # Full benchmark including mlx_lm baseline:
  python examples/bench_compare.py
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_VENV = os.path.join(_ROOT, ".venv")
_PYTHON = os.path.join(_VENV, "bin", "python")

BASE = "http://localhost:8080"
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
N_RUNS = 5  # statistical runs per scenario

PROMPTS = {
    "short":  ("What is 2+2?", 20),
    "medium": ("Explain how the TCP/IP three-way handshake works.", 120),
    "long":   (
        "Write a Python function that implements merge sort. "
        "Include detailed comments explaining each step and the time complexity.",
        250,
    ),
}

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(method, path, body=None, timeout=300):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(path):    return _req("GET", path)
def post(path, body): return _req("POST", path, body)
def delete(path): return _req("DELETE", path)


def stream_chunks(path, body, timeout=300):
    """Yield (chunk_dict, elapsed_secs) for each SSE data frame."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read(128)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                line = line.strip()
                if line.startswith(b"data:"):
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        return
                    try:
                        yield json.loads(payload), time.perf_counter() - t0
                    except Exception:
                        pass


def wait_ready(base=BASE, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False

# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(port=8080, extra_env=None):
    binary = os.path.join(_ROOT, "target/release/mlx-lm-server")
    if not os.path.exists(binary):
        return None
    env = {
        **os.environ,
        "MLX_PORT": str(port),
        "MLX_DEBUG": "false",
        "PYTHONPATH": f"{_VENV}/lib/python3.13/site-packages",
        "VIRTUAL_ENV": _VENV,
        **(extra_env or {}),
    }
    proc = subprocess.Popen(
        [binary], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def proc_mem_mb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip()) / 1024
    except Exception:
        return 0

# ── Core measurements ─────────────────────────────────────────────────────────

def measure_model_load(model=MODEL):
    """POST /v1/models/load and return seconds taken."""
    t0 = time.perf_counter()
    post("/v1/models/load", {"model": model})
    return time.perf_counter() - t0


def measure_prefill_decode(prompt, max_tokens, kv_bits=None):
    """
    Returns dict with keys:
      ttft_ms, total_ms, prefill_tps, decode_tps,
      prompt_tokens, gen_tokens
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if kv_bits is not None:
        body["kv_bits"] = kv_bits

    t_first = None
    t_last = None
    token_count = 0

    for chunk, elapsed in stream_chunks("/v1/chat/completions", body):
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                if t_first is None:
                    t_first = elapsed
                t_last = elapsed
                token_count += 1

    # Non-streaming to get accurate token counts and total wall time
    body_ns = {**body, "stream": False}
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", body_ns)
    total_wall = time.perf_counter() - t0
    usage = resp.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    gen_tokens = usage.get("completion_tokens", 0)

    if t_first is None or t_last is None:
        return None

    ttft = t_first
    decode_time = t_last - t_first
    prefill_tps = prompt_tokens / ttft if ttft > 0 else 0
    decode_tps = gen_tokens / decode_time if decode_time > 0 else 0

    return {
        "ttft_ms": ttft * 1000,
        "total_ms": total_wall * 1000,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
    }


def run_scenario(label, prompt, max_tokens, n=N_RUNS, kv_bits=None):
    """Run N_RUNS measurements and return aggregated stats."""
    results = []
    for i in range(n):
        r = measure_prefill_decode(prompt, max_tokens, kv_bits=kv_bits)
        if r:
            results.append(r)

    if not results:
        return None

    def pct(key, p):
        vals = sorted(r[key] for r in results)
        idx = int(math.ceil(p / 100.0 * len(vals))) - 1
        return vals[max(0, idx)]

    return {
        "label": label,
        "n": len(results),
        "kv_bits": kv_bits,
        "ttft_ms_p50":     pct("ttft_ms", 50),
        "ttft_ms_p95":     pct("ttft_ms", 95),
        "prefill_tps_avg": statistics.mean(r["prefill_tps"] for r in results),
        "decode_tps_avg":  statistics.mean(r["decode_tps"] for r in results),
        "decode_tps_p50":  pct("decode_tps", 50),
        "decode_tps_p95":  pct("decode_tps", 95),
        "total_ms_avg":    statistics.mean(r["total_ms"] for r in results),
        "prompt_tokens":   results[0]["prompt_tokens"],
        "gen_tokens":      results[0]["gen_tokens"],
    }


def bench_concurrent(n_concurrent=4, max_tokens=80):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    latencies, errors = [], []
    lock = threading.Lock()

    def worker():
        t0 = time.perf_counter()
        try:
            post("/v1/chat/completions", body)
            with lock:
                latencies.append(time.perf_counter() - t0)
        except Exception as e:
            with lock:
                errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(n_concurrent)]
    t_wall = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t_wall
    return latencies, wall, errors


# ── mlx_lm baseline ──────────────────────────────────────────────────────────

_BASELINE_SCRIPT = (
    "import sys, time, mlx_lm\n"
    "model_id = sys.argv[1]\n"
    "prompt   = sys.argv[2]\n"
    "max_tok  = int(sys.argv[3])\n"
    "t0 = time.perf_counter()\n"
    "model, tokenizer = mlx_lm.load(model_id)\n"
    "load_s = time.perf_counter() - t0\n"
    "responses = list(mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=max_tok))\n"
    "last = responses[-1]\n"
    "print(f'LOAD_S={load_s:.3f}')\n"
    "print(f'PROMPT_TPS={last.prompt_tps:.2f}')\n"
    "print(f'GEN_TPS={last.generation_tps:.2f}')\n"
    "print(f'PROMPT_TOKENS={last.prompt_tokens}')\n"
    "print(f'GEN_TOKENS={last.generation_tokens}')\n"
)


def run_baseline(prompt, max_tokens):
    """Run mlx_lm.stream_generate directly and parse tps metrics."""
    try:
        result = subprocess.run(
            [_PYTHON, "-c", _BASELINE_SCRIPT, MODEL, prompt, str(max_tokens)],
            capture_output=True, text=True, timeout=300,
        )
        out = result.stdout
        def parse(key):
            m = re.search(rf"{key}=([0-9.]+)", out)
            return float(m.group(1)) if m else None

        return {
            "load_s":       parse("LOAD_S"),
            "prompt_tps":   parse("PROMPT_TPS"),
            "gen_tps":      parse("GEN_TPS"),
            "prompt_tokens": parse("PROMPT_TOKENS"),
            "gen_tokens":   parse("GEN_TOKENS"),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Reporting ─────────────────────────────────────────────────────────────────

def hdr(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def print_scenario(s):
    kv_tag = f" [kv_bits={s['kv_bits']}]" if s.get("kv_bits") else ""
    print(f"\n  [{s['label']}{kv_tag}]  {s['prompt_tokens']} prompt → {s['gen_tokens']} gen tokens")
    print(f"    TTFT:         p50={s['ttft_ms_p50']:.0f} ms   p95={s['ttft_ms_p95']:.0f} ms")
    print(f"    Prefill TPS:  {s['prefill_tps_avg']:.1f} tok/s")
    print(f"    Decode TPS:   avg={s['decode_tps_avg']:.1f}  p50={s['decode_tps_p50']:.1f}  p95={s['decode_tps_p95']:.1f} tok/s")
    print(f"    Wall time:    {s['total_ms_avg']:.0f} ms avg")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global BASE, MODEL, N_RUNS  # noqa: PLW0603

    ap = argparse.ArgumentParser()
    ap.add_argument("--server-only", action="store_true",
                    help="Skip mlx_lm baseline (server must already be running)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="Test with KV quantization (4 or 8)")
    ap.add_argument("--no-kv", action="store_true",
                    help="Skip KV-quantized comparison run")
    args = ap.parse_args()

    BASE = f"http://localhost:{args.port}"
    MODEL = args.model
    N_RUNS = args.runs

    managed_proc = None
    print(f"{'='*60}")
    print("  mlx-lm-server vs mlx_lm baseline benchmark")
    print(f"  Model:   {MODEL}")
    print(f"  Runs:    {N_RUNS} per scenario")
    print(f"{'='*60}")

    # ── Server startup ──────────────────────────────────────────────────────
    if not wait_ready(timeout=3):
        print("\nServer not detected — starting it now...")
        t0 = time.perf_counter()
        managed_proc = start_server(port=args.port)
        if not managed_proc:
            print("  ERROR: binary not found; build first with: cargo build -p mlx-lm-server --release")
            sys.exit(1)
        if not wait_ready(timeout=60):
            print("  ERROR: server did not become healthy in 60 s"); sys.exit(1)
        startup_s = time.perf_counter() - t0
        idle_mem = proc_mem_mb(managed_proc.pid)
        print(f"  Cold start:  {startup_s*1000:.0f} ms")
        print(f"  Idle RSS:    {idle_mem:.0f} MB")
    else:
        print("  Using already-running server.")

    # ── Model load ──────────────────────────────────────────────────────────
    hdr("1. Model load")
    load_s = measure_model_load(MODEL)
    print(f"  Load time: {load_s:.2f} s")
    if managed_proc:
        loaded_mem = proc_mem_mb(managed_proc.pid)
        print(f"  RSS after load: {loaded_mem:.0f} MB")

    # ── Per-prompt scenarios (baseline kv_bits=None) ─────────────────────────
    hdr("2. Prefill TPS  /  Decode TPS  (no KV quant)")
    baseline_results = {}
    for key, (prompt, max_tok) in PROMPTS.items():
        s = run_scenario(key, prompt, max_tok, n=N_RUNS, kv_bits=None)
        if s:
            baseline_results[key] = s
            print_scenario(s)

    # ── KV quantization comparison ───────────────────────────────────────────
    if not args.no_kv:
        for bits in ([args.kv_bits] if args.kv_bits else [4, 8]):
            hdr(f"3. KV-quantized decode  (kv_bits={bits})")
            print("  NOTE: accuracy may differ slightly from baseline (quantization noise).")
            for key, (prompt, max_tok) in PROMPTS.items():
                s = run_scenario(key, prompt, max_tok, n=max(2, N_RUNS // 2), kv_bits=bits)
                if s:
                    base = baseline_results.get(key)
                    speedup = ""
                    if base and base["decode_tps_avg"] > 0:
                        ratio = s["decode_tps_avg"] / base["decode_tps_avg"]
                        speedup = f"  ({ratio:+.0%} vs baseline)"
                    print_scenario(s)
                    print(f"    {speedup}")

    # ── Concurrent load ──────────────────────────────────────────────────────
    hdr("4. Concurrency (4 parallel requests)")
    lats, wall, errs = bench_concurrent(n_concurrent=4, max_tokens=80)
    if lats:
        lats.sort()
        print(f"  Wall time:   {wall:.2f}s")
        print(f"  Avg latency: {statistics.mean(lats):.2f}s")
        print(f"  p50 latency: {lats[len(lats)//2]:.2f}s")
        p95_idx = int(math.ceil(0.95 * len(lats))) - 1
        print(f"  p95 latency: {lats[p95_idx]:.2f}s")
        print(f"  Errors: {len(errs)}")

    # ── mlx_lm baseline comparison ──────────────────────────────────────────
    if not args.server_only:
        hdr("5. mlx_lm raw baseline (subprocess, fresh model load)")
        print("  (This runs mlx_lm.stream_generate directly — no HTTP overhead)")
        for key, (prompt, max_tok) in PROMPTS.items():
            print(f"\n  [{key}]", flush=True)
            b = run_baseline(prompt, max_tok)
            if "error" in b:
                print(f"    ERROR: {b['error']}")
            else:
                print(f"    Load:         {b['load_s']:.2f} s")
                print(f"    Prefill TPS:  {b['prompt_tps']:.1f} tok/s")
                print(f"    Decode TPS:   {b['gen_tps']:.1f} tok/s")
                srv = baseline_results.get(key)
                if srv:
                    p_ratio = srv["prefill_tps_avg"] / b["prompt_tps"] if b["prompt_tps"] else 0
                    d_ratio = srv["decode_tps_avg"] / b["gen_tps"] if b["gen_tps"] else 0
                    print(f"    Server overhead: prefill {p_ratio:.2f}x  decode {d_ratio:.2f}x")
                    print(f"    (1.0x = same speed; <1.0x = server is slower)")

    # ── Summary table ────────────────────────────────────────────────────────
    hdr("SUMMARY")
    print(f"  {'Prompt':<8} {'Prefill TPS':>12} {'Decode TPS p50':>15} {'TTFT p50 ms':>13}")
    print(f"  {'─'*8} {'─'*12} {'─'*15} {'─'*13}")
    for key, s in baseline_results.items():
        print(f"  {key:<8} {s['prefill_tps_avg']:>12.1f} {s['decode_tps_p50']:>15.1f} {s['ttft_ms_p50']:>13.0f}")

    print(f"\n{'='*60}")
    print("  Tips to improve performance:")
    print("    MLX_PREFILL_STEP_SIZE=2048   faster prefill for long prompts")
    print("    MLX_DEFAULT_KV_BITS=4        ~2x KV memory reduction (slight quality trade-off)")
    print("    MLX_QUANTIZED_KV_START=1000  quantize KV only after 1000 tokens")
    print("    MLX_MOE_TOP_K=4              7-16% decode speedup on MoE models")
    print("    MLX_METAL_CACHE_LIMIT_GB=2   more Metal kernel cache on ≥16 GB machines")
    print(f"{'='*60}")

    if managed_proc:
        managed_proc.terminate()
        managed_proc.wait()


if __name__ == "__main__":
    main()
