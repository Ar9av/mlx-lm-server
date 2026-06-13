#!/usr/bin/env python3
"""
Multi-model benchmark for mlx-lm-server.
Tests every locally-cached LLM across short/medium/long prompts.
Outputs structured JSON + pretty-printed table.

Usage:
  python examples/bench_all_models.py                    # baseline
  python examples/bench_all_models.py --tag v2-prefill  # label a run
  python examples/bench_all_models.py --compare a.json b.json
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
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_VENV = os.path.join(_ROOT, ".venv")
_BINARY = os.path.join(_ROOT, "target/release/mlx-lm-server")

BASE_PORT = 9988          # use dedicated port so we don't clash with dev server
BASE = f"http://localhost:{BASE_PORT}"
N_RUNS = 3                # stat runs per scenario

LLM_MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
    # Mistral-7B: weights not locally cached — excluded from offline benchmark
]

# A ~480-token context for prefill-throughput measurement.
# Pasting a real document excerpt stresses the prefill path more than short prompts.
_LONG_CTX = (
    "The following is an excerpt from a technical specification on distributed systems.\n\n"
    "In a distributed computing environment, consistency and availability are two fundamental "
    "properties that system designers must balance. The CAP theorem, formulated by Eric Brewer "
    "in 2000 and formally proven by Gilbert and Lynch in 2002, states that a distributed data "
    "store can provide at most two of the following three guarantees simultaneously: consistency "
    "(every read receives the most recent write or an error), availability (every request "
    "receives a non-error response, without the guarantee that it contains the most recent write), "
    "and partition tolerance (the system continues to operate despite an arbitrary number of "
    "messages being dropped by the network between nodes).\n\n"
    "In practice, network partitions are an unavoidable reality in distributed systems, meaning "
    "that engineers must choose between consistency and availability when a partition occurs. "
    "Distributed databases like Apache Cassandra, Amazon DynamoDB, and Riak prioritize "
    "availability (AP systems), while systems like Apache ZooKeeper, etcd, and Google Spanner "
    "prioritize consistency (CP systems).\n\n"
    "Modern distributed systems often implement eventual consistency, a model in which updates "
    "propagate through the system over time, and all replicas eventually converge to the same "
    "value in the absence of further updates. Vector clocks, Lamport timestamps, and CRDTs "
    "(Conflict-free Replicated Data Types) are common mechanisms for tracking causal "
    "relationships and resolving conflicts in eventually consistent systems.\n\n"
    "Question: In one sentence, what is the CAP theorem?"
)

PROMPTS = {
    # Token budgets are generous to accommodate thinking models (Qwen3) that
    # use additional tokens for chain-of-thought before the actual answer.
    "short":  ("What is the capital of France?", 200),
    "medium": ("Explain how the TCP/IP three-way handshake works in simple terms.", 400),
    "long":   (
        "Write a Python function that implements merge sort with detailed inline "
        "comments explaining each step. Include time and space complexity analysis.",
        700,
    ),
    # Stresses prefill throughput: ~480-token prompt, short output
    "prefill": (_LONG_CTX, 80),
}

# ── HTTP ──────────────────────────────────────────────────────────────────────

def _req(method, path, body=None, timeout=300):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    hdr = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(p):       return _req("GET", p)
def post(p, b):   return _req("POST", p, b)


def stream_chunks(path, body, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read(256)
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


def wait_ready(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.4)
    return False

# ── Process helpers ───────────────────────────────────────────────────────────

def proc_mem_mb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip()) / 1024
    except Exception:
        return 0

# ── Core measure ──────────────────────────────────────────────────────────────

def _build_messages(prompt):
    """Wrap prompt with a brief system message that also helps thinking models stay concise."""
    return [
        {"role": "system", "content": "Be concise and direct. Answer in as few words as needed."},
        {"role": "user", "content": prompt},
    ]


def one_streaming_pass(model_id, prompt, max_tokens, kv_bits=None):
    """
    Returns (ttft_s, decode_s, stream_tokens) or None on failure.
    ttft_s  = time from request start to first content token
    decode_s = time from first content token to last
    """
    body = {
        "model": model_id,
        "messages": _build_messages(prompt),
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    if kv_bits:
        body["kv_bits"] = kv_bits

    t_first = None
    t_last  = None
    n_tok   = 0
    try:
        for chunk, elapsed in stream_chunks("/v1/chat/completions", body):
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                if t_first is None:
                    t_first = elapsed
                t_last = elapsed
                n_tok += 1
    except Exception:
        return None

    if t_first is None:
        return None
    return t_first, max(t_last - t_first, 1e-4), n_tok


def one_non_stream_pass(model_id, prompt, max_tokens, kv_bits=None):
    """Returns (wall_s, prompt_tokens, gen_tokens) or None."""
    body = {
        "model": model_id,
        "messages": _build_messages(prompt),
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    if kv_bits:
        body["kv_bits"] = kv_bits
    try:
        t0 = time.perf_counter()
        resp = post("/v1/chat/completions", body)
        wall = time.perf_counter() - t0
        usage = resp.get("usage", {})
        return wall, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    except Exception:
        return None


def bench_scenario(model_id, prompt_key, prompt, max_tokens, n=N_RUNS, kv_bits=None):
    """
    n streaming + n non-streaming passes; returns aggregated stats dict.
    """
    stream_results = []
    ns_results = []

    for _ in range(n):
        r = one_streaming_pass(model_id, prompt, max_tokens, kv_bits)
        if r:
            stream_results.append(r)
        ns = one_non_stream_pass(model_id, prompt, max_tokens, kv_bits)
        if ns:
            ns_results.append(ns)

    if not stream_results or not ns_results:
        return None

    # Use non-streaming for accurate token counts
    prompt_tokens = statistics.median(r[1] for r in ns_results)
    gen_tokens    = statistics.median(r[2] for r in ns_results)

    ttfts    = [r[0] for r in stream_results]
    decodes  = [r[1] for r in stream_results]
    n_toks   = [r[2] for r in stream_results]

    # prefill TPS from TTFT and prompt token count
    prefill_tps_list = [prompt_tokens / t for t in ttfts if t > 0]
    # decode TPS from decode window and actual generated tokens
    decode_tps_list  = [nt / d for nt, d in zip(n_toks, decodes) if d > 0]

    def pct(lst, p):
        s = sorted(lst)
        idx = max(0, int(math.ceil(p / 100 * len(s))) - 1)
        return s[idx]

    return {
        "prompt_key":     prompt_key,
        "kv_bits":        kv_bits,
        "prompt_tokens":  int(prompt_tokens),
        "gen_tokens":     int(gen_tokens),
        "n_runs":         len(stream_results),
        "ttft_ms_p50":    round(pct(ttfts, 50) * 1000, 1),
        "ttft_ms_p95":    round(pct(ttfts, 95) * 1000, 1),
        "prefill_tps":    round(statistics.mean(prefill_tps_list), 1),
        "decode_tps_avg": round(statistics.mean(decode_tps_list), 1),
        "decode_tps_p50": round(pct(decode_tps_list, 50), 1),
        "decode_tps_p95": round(pct(decode_tps_list, 95), 1),
        "wall_ms_avg":    round(statistics.mean(r[0] for r in ns_results) * 1000, 0),
    }


def bench_concurrent(model_id, n=4, max_tokens=60):
    body = {
        "model": model_id,
        "messages": _build_messages("What is 2+2?"),
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
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

    threads = [threading.Thread(target=worker) for _ in range(n)]
    t_wall = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t_wall

    if not latencies:
        return None
    return {
        "n": n,
        "wall_s":    round(wall, 3),
        "avg_lat_s": round(statistics.mean(latencies), 3),
        "p50_lat_s": round(sorted(latencies)[len(latencies)//2], 3),
        "errors":    len(errors),
    }

# ── Per-model benchmark ───────────────────────────────────────────────────────

def bench_model(model_id, kv_bits=None, extra_env=None):
    print(f"  Loading {model_id} ...", flush=True)
    t0 = time.perf_counter()
    try:
        post("/v1/models/load", {"model": model_id})
    except Exception as e:
        print(f"    LOAD FAILED: {e}")
        return None
    load_s = time.perf_counter() - t0
    print(f"  Loaded in {load_s:.1f}s", flush=True)

    results = {"model": model_id, "load_s": round(load_s, 2),
               "kv_bits": kv_bits, "scenarios": []}

    for key, (prompt, max_tok) in PROMPTS.items():
        print(f"    [{key}] ...", end="", flush=True)
        s = bench_scenario(model_id, key, prompt, max_tok, kv_bits=kv_bits)
        if s:
            results["scenarios"].append(s)
            print(f" decode={s['decode_tps_avg']:.0f} tok/s  prefill={s['prefill_tps']:.0f} tok/s  TTFT={s['ttft_ms_p50']:.0f}ms")
        else:
            print(" FAILED")

    # Concurrency test on short prompt only
    print(f"    [concurrent-4] ...", end="", flush=True)
    cc = bench_concurrent(model_id, n=4, max_tokens=60)
    if cc:
        results["concurrent"] = cc
        print(f" wall={cc['wall_s']:.2f}s  avg_lat={cc['avg_lat_s']:.2f}s")

    # Unload to free memory for next model
    short_name = model_id.split("/")[-1]
    try:
        _req("DELETE", f"/v1/models/{urllib.parse.quote(model_id, safe='')}")
    except Exception:
        pass

    return results

# ── Comparison printing ───────────────────────────────────────────────────────

def print_table(run):
    tag = run.get("tag", "?")
    env = run.get("env", {})
    print(f"\n{'='*72}")
    print(f"  Run: {tag}")
    if env:
        print(f"  Env: {env}")
    print(f"{'='*72}")
    print(f"  {'Model':<32} {'Prompt':<8} {'Prefill':>8} {'Decode':>8} {'TTFT':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in run.get("models", []):
        model_short = m["model"].split("/")[-1][:30]
        for s in m.get("scenarios", []):
            print(f"  {model_short:<32} {s['prompt_key']:<8} "
                  f"{s['prefill_tps']:>7.0f}T {s['decode_tps_avg']:>7.0f}T "
                  f"{s['ttft_ms_p50']:>7.0f}ms")
    print()


def print_comparison(base, test):
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {base.get('tag','baseline')} → {test.get('tag','test')}")
    print(f"{'='*80}")
    print(f"  {'Model':<30} {'Prompt':<8} {'Prefill Δ':>10} {'Decode Δ':>10} {'TTFT Δ':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    base_idx = {(m["model"], s["prompt_key"]): s
                for m in base.get("models", []) for s in m.get("scenarios", [])}
    for m in test.get("models", []):
        for s in m.get("scenarios", []):
            key = (m["model"], s["prompt_key"])
            b = base_idx.get(key)
            if not b:
                continue
            p_delta = (s["prefill_tps"] - b["prefill_tps"]) / max(b["prefill_tps"], 1) * 100
            d_delta = (s["decode_tps_avg"] - b["decode_tps_avg"]) / max(b["decode_tps_avg"], 1) * 100
            t_delta = (s["ttft_ms_p50"] - b["ttft_ms_p50"]) / max(b["ttft_ms_p50"], 1) * 100
            mshort = m["model"].split("/")[-1][:28]
            print(f"  {mshort:<30} {s['prompt_key']:<8} "
                  f"{p_delta:>+9.1f}% {d_delta:>+9.1f}% {t_delta:>+9.1f}%")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global N_RUNS  # noqa: PLW0603
    import urllib.parse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline", help="Label for this run")
    ap.add_argument("--out", default=None, help="Output JSON path (default: bench_<tag>.json)")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "TEST"),
                    help="Compare two JSON result files")
    ap.add_argument("--models", nargs="*", default=LLM_MODELS,
                    help="Models to benchmark (default: all)")
    ap.add_argument("--kv-bits", type=int, default=None)
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--skip-start", action="store_true",
                    help="Assume server is already running on port 9988")
    args = ap.parse_args()

    N_RUNS = args.runs

    # ── Compare mode ────────────────────────────────────────────────────────
    if args.compare:
        with open(args.compare[0]) as f: base = json.load(f)
        with open(args.compare[1]) as f: test = json.load(f)
        print_comparison(base, test)
        return

    # ── Start server ────────────────────────────────────────────────────────
    managed_proc = None
    if not args.skip_start:
        if not os.path.exists(_BINARY):
            print(f"ERROR: binary not found at {_BINARY}")
            print("Build with: cargo build -p mlx-lm-server --release")
            sys.exit(1)

        env = {
            **os.environ,
            "MLX_PORT": str(BASE_PORT),
            "MLX_DEBUG": "false",
            "MLX_KEEP_ALIVE_SECS": "3600",
            "PYTHONPATH": f"{_VENV}/lib/python3.13/site-packages",
            "VIRTUAL_ENV": _VENV,
        }
        print(f"Starting server on port {BASE_PORT}...")
        managed_proc = subprocess.Popen(
            [_BINARY], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not wait_ready(timeout=60):
            print("ERROR: server did not become healthy")
            managed_proc.kill()
            sys.exit(1)
        idle_mem = proc_mem_mb(managed_proc.pid)
        print(f"Server ready. Idle RSS: {idle_mem:.0f} MB")
    else:
        if not wait_ready(timeout=5):
            print(f"ERROR: no server on port {BASE_PORT}")
            sys.exit(1)
        print("Using existing server.")

    # ── Benchmark each model ────────────────────────────────────────────────
    run = {
        "tag": args.tag,
        "ts":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kv_bits": args.kv_bits,
        "n_runs": N_RUNS,
        "models": [],
    }

    print(f"\n{'='*60}")
    print(f"  Benchmarking {len(args.models)} models  (tag={args.tag})")
    print(f"{'='*60}")

    for model_id in args.models:
        print(f"\n[{model_id}]")
        r = bench_model(model_id, kv_bits=args.kv_bits)
        if r:
            run["models"].append(r)

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = args.out or f"bench_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_table(run)

    if managed_proc:
        managed_proc.terminate()
        managed_proc.wait()


if __name__ == "__main__":
    main()
