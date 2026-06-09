#!/usr/bin/env python3
"""Benchmark script for mlx-lm-server."""

import time
import json
import subprocess
import os
import sys
import statistics
import threading
import urllib.request
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

BASE = "http://localhost:8765"
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
PROMPTS = {
    "short":  "What is 2+2?",
    "medium": "Explain how the TCP/IP handshake works in simple terms.",
    "long":   "Write a Python function that implements merge sort and explain each step of the algorithm in detail.",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

def post_stream(path, body):
    """Yields (chunk_json, elapsed_since_start) for each SSE chunk."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data,
                                  headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        buf = b""
        while True:
            chunk = r.read(64)
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
                        yield json.loads(payload), time.perf_counter() - start
                    except Exception:
                        pass

def proc_mem_mb(pid):
    """RSS in MB on macOS via ps."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip()) / 1024
    except Exception:
        return 0

def wait_ready(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            get("/health")
            return True
        except Exception:
            time.sleep(0.2)
    return False

# ── benchmarks ────────────────────────────────────────────────────────────────

def bench_startup(binary, env):
    """Cold-start: time from process launch to first /health 200."""
    t0 = time.perf_counter()
    proc = subprocess.Popen([binary], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    ok = wait_ready()
    elapsed = time.perf_counter() - t0
    if not ok:
        proc.kill()
        return None, None
    return proc, elapsed

def bench_ttft(label, prompt, max_tokens=200):
    """Time to first token via streaming."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    results = []
    for _ in range(3):
        for chunk, elapsed in post_stream("/v1/chat/completions", body):
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                results.append(elapsed)
                break
        try:
            for _ in post_stream("/v1/chat/completions", body):
                pass
        except Exception:
            pass
    return statistics.mean(results) if results else None

def bench_throughput(label, prompt, max_tokens=200):
    """Tokens/sec and total generation time (non-streaming)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        resp = post("/v1/chat/completions", body)
        elapsed = time.perf_counter() - t0
        tokens = resp.get("usage", {}).get("completion_tokens", 0)
        if tokens > 0:
            runs.append((elapsed, tokens, tokens / elapsed))
    if not runs:
        return None
    avg_tps = statistics.mean(r[2] for r in runs)
    avg_elapsed = statistics.mean(r[0] for r in runs)
    avg_tokens = statistics.mean(r[1] for r in runs)
    return avg_elapsed, avg_tokens, avg_tps

def bench_stream_throughput(label, prompt, max_tokens=200):
    """Measure streaming throughput: tokens/sec from first to last token."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    runs = []
    for _ in range(3):
        first_t = None
        last_t = None
        token_count = 0
        for chunk, elapsed in post_stream("/v1/chat/completions", body):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    if first_t is None:
                        first_t = elapsed
                    last_t = elapsed
                    token_count += 1
        if first_t and last_t and token_count > 1:
            span = last_t - first_t
            if span > 0:
                runs.append((first_t, last_t, token_count, token_count / span))
    if not runs:
        return None
    return (
        statistics.mean(r[0] for r in runs),
        statistics.mean(r[3] for r in runs),
        statistics.mean(r[2] for r in runs),
    )

def bench_concurrent(n_concurrent=4, max_tokens=80):
    """Latency under N concurrent requests."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    latencies = []
    errors = []
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
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_wall

    return latencies, wall, errors

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    binary = os.path.join(_ROOT, "target/release/mlx-lm-server")
    venv = os.path.join(_ROOT, ".venv")
    env = {
        **os.environ,
        "MLX_PORT": "8765",
        "MLX_DEBUG": "false",
        "PYTHONPATH": f"{venv}/lib/python3.13/site-packages",
        "VIRTUAL_ENV": venv,
    }

    print("=" * 60)
    print("  mlx-lm-server benchmark")
    print("=" * 60)

    print("\n[1/5] Cold-start latency...")
    proc, startup_s = bench_startup(binary, env)
    if proc is None:
        print("  FAILED to start server"); sys.exit(1)
    idle_mem = proc_mem_mb(proc.pid)
    print(f"  Cold start:  {startup_s*1000:.0f} ms")
    print(f"  Idle RSS:    {idle_mem:.1f} MB")

    print(f"\n[2/5] Model load: {MODEL} ...")
    t0 = time.perf_counter()
    post("/v1/models/load", {"model": MODEL})
    load_s = time.perf_counter() - t0
    loaded_mem = proc_mem_mb(proc.pid)
    print(f"  Model load:  {load_s:.2f} s")
    print(f"  Loaded RSS:  {loaded_mem:.1f} MB")
    print(f"  Model delta: {loaded_mem - idle_mem:.1f} MB")

    print("\n[3/5] Non-streaming throughput (3 runs each)...")
    for key, prompt in PROMPTS.items():
        r = bench_throughput(key, prompt)
        if r:
            elapsed, tokens, tps = r
            print(f"  {key:6s}: {tps:6.1f} tok/s  |  {elapsed:.2f}s  |  {tokens:.0f} tokens")
    peak_mem = proc_mem_mb(proc.pid)

    print("\n[4/5] Streaming: TTFT + throughput (3 runs each)...")
    for key, prompt in PROMPTS.items():
        r = bench_stream_throughput(key, prompt)
        if r:
            ttft, tps, tokens = r
            print(f"  {key:6s}: TTFT {ttft*1000:.0f} ms  |  {tps:.1f} tok/s  |  {tokens:.0f} tokens")

    peak_mem2 = proc_mem_mb(proc.pid)
    print(f"\n  Peak RSS during inference: {max(peak_mem, peak_mem2):.1f} MB")

    print("\n[5/5] Concurrency: 4 parallel requests...")
    lats, wall, errs = bench_concurrent(n_concurrent=4, max_tokens=80)
    if lats:
        print(f"  Wall time:   {wall:.2f}s")
        print(f"  Avg latency: {statistics.mean(lats):.2f}s")
        print(f"  Min/Max:     {min(lats):.2f}s / {max(lats):.2f}s")
        print(f"  Errors:      {len(errs)}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    r_short = bench_throughput("short", PROMPTS["short"])
    r_medium = bench_throughput("medium", PROMPTS["medium"])
    print(f"  Server cold-start:          {startup_s*1000:.0f} ms")
    print(f"  Model load (Llama-3.2-1B):  {load_s:.2f} s (cached)")
    print(f"  Idle memory:                {idle_mem:.0f} MB RSS")
    print(f"  Loaded memory:              {loaded_mem:.0f} MB RSS")
    print(f"  Peak inference memory:      {max(peak_mem, peak_mem2):.0f} MB RSS")
    if r_short:
        print(f"  Throughput (short prompt):  {r_short[2]:.1f} tok/s")
    if r_medium:
        print(f"  Throughput (medium prompt): {r_medium[2]:.1f} tok/s")
    stream_short = bench_stream_throughput("short", PROMPTS["short"])
    if stream_short:
        print(f"  TTFT (short prompt):        {stream_short[0]*1000:.0f} ms")
    print("=" * 60)

    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()
