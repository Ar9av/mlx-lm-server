#!/usr/bin/env python3
"""
Speculative decoding benchmark.
Loads target model with a draft model and compares tok/s vs solo generation.
Best results for code/structured text (high acceptance rates).
"""
import json
import math
import statistics
import sys
import time
import urllib.request
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_VENV = os.path.join(_ROOT, ".venv")
_BINARY = os.path.join(_ROOT, "target/release/mlx-lm-server")

PORT = 9989
BASE = f"http://localhost:{PORT}"

TARGET_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
DRAFT_MODEL  = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# High-acceptance-rate prompts: code and structured output
CODE_PROMPTS = [
    ("Write a Python class that implements a binary search tree with insert, search, and delete methods. Add docstrings.", 400),
    ("Write a recursive function to compute Fibonacci numbers with memoization in Python.", 200),
    ("Write a Python function that parses a CSV string into a list of dictionaries, handling quoted fields.", 300),
]

def _req(method, path, body=None, timeout=300):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    hdr = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def post(p, b): return _req("POST", p, b)
def wait_ready(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.4)
    return False

def bench_decode_tps(prompt, max_tokens, n=3):
    """Return (decode_tps_avg, decode_tps_p50, wall_s) over N runs."""
    results = []
    for _ in range(n):
        t0 = time.perf_counter()
        resp = post("/v1/chat/completions", {
            "model": TARGET_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert Python programmer. Write clean, efficient code."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.0,
        })
        wall = time.perf_counter() - t0
        usage = resp.get("usage", {})
        gen_tok = usage.get("completion_tokens", 0)
        if gen_tok > 5:
            results.append((wall, gen_tok, gen_tok / wall))
    if not results:
        return None
    return (
        statistics.mean(r[2] for r in results),
        sorted(results, key=lambda x: x[2])[len(results)//2][2],
        statistics.mean(r[0] for r in results),
        statistics.mean(r[1] for r in results),
    )

def main():
    # Start server
    env = {
        **os.environ,
        "MLX_PORT": str(PORT),
        "MLX_DEBUG": "false",
        "MLX_KEEP_ALIVE_SECS": "3600",
        "PYTHONPATH": f"{_VENV}/lib/python3.13/site-packages",
        "VIRTUAL_ENV": _VENV,
    }
    proc = subprocess.Popen([_BINARY], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_ready():
        print("Server failed to start")
        proc.kill(); sys.exit(1)
    print(f"Server ready on :{PORT}")

    print(f"\nTarget: {TARGET_MODEL}")
    print(f"Draft:  {DRAFT_MODEL}")
    print(f"{'='*60}")

    # ── Solo generation (no draft) ────────────────────────────────
    print("\n[1/2] Solo generation (no speculative decoding)...")
    post("/v1/models/load", {"model": TARGET_MODEL})
    print(f"  Loaded target model.")

    solo_results = []
    for prompt, max_tok in CODE_PROMPTS:
        r = bench_decode_tps(prompt, max_tok)
        if r:
            avg, p50, wall, ntok = r
            print(f"  {prompt[:50]!r:52s}  {avg:.0f} tok/s  {wall:.1f}s  {ntok:.0f}tok")
            solo_results.append(avg)

    # ── Speculative decoding ────────────────────────────────────
    print(f"\n[2/2] Speculative decoding (draft={DRAFT_MODEL.split('/')[-1]}, num_draft_tokens=4)...")
    post("/v1/models/load", {"model": TARGET_MODEL, "drafter": DRAFT_MODEL})
    print(f"  Loaded target + draft model.")

    spec_results = []
    for prompt, max_tok in CODE_PROMPTS:
        r = bench_decode_tps(prompt, max_tok)
        if r:
            avg, p50, wall, ntok = r
            print(f"  {prompt[:50]!r:52s}  {avg:.0f} tok/s  {wall:.1f}s  {ntok:.0f}tok")
            spec_results.append(avg)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SPECULATIVE DECODING SUMMARY")
    print(f"{'='*60}")
    if solo_results and spec_results:
        solo_avg = statistics.mean(solo_results)
        spec_avg = statistics.mean(spec_results)
        ratio = spec_avg / solo_avg
        print(f"  Solo avg decode:       {solo_avg:.0f} tok/s")
        print(f"  Speculative avg decode:{spec_avg:.0f} tok/s")
        print(f"  Speedup:               {ratio:.2f}x  ({(ratio-1)*100:+.0f}%)")
        if ratio > 1.0:
            print("  → Speculative decoding HELPS for this model pair / task.")
        else:
            print("  → Speculative decoding HURTS (draft overhead > acceptance gain).")

    # Save results
    out = {
        "target": TARGET_MODEL,
        "draft": DRAFT_MODEL,
        "solo_tps": solo_results,
        "spec_tps": spec_results,
        "speedup": (statistics.mean(spec_results)/statistics.mean(solo_results)) if (solo_results and spec_results) else None,
    }
    with open("bench_speculative.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to bench_speculative.json")

    proc.terminate(); proc.wait()

if __name__ == "__main__":
    main()
