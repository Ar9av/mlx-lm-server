#!/usr/bin/env python3
"""
Live STS demo — record from mic (+ optional background mix), separate, play back.

SAM-Audio needs MIXED audio to separate anything meaningful.
If you record in a silent room, there's nothing to pull apart.

Best demo: play music from speakers while you talk, then run this —
it should extract your voice in target and leave the music in residual.

Usage:
  python sts_demo.py                          # record 5s, description "speech"
  python sts_demo.py --duration 8             # record 8s
  python sts_demo.py --description "music"    # extract music instead
  python sts_demo.py --mix background.wav     # mix a background track into recording
"""
import argparse
import base64
import io
import sys
import time

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# SAM-Audio native sample rate
MODEL_SR = 48000
CHANNELS = 1


def record(seconds: float) -> np.ndarray:
    """Record from mic at MODEL_SR, return float32 mono array."""
    print(f"  🎙  Recording {seconds}s — speak now (play background music for best results)!")
    frames = sd.rec(int(seconds * MODEL_SR), samplerate=MODEL_SR,
                    channels=CHANNELS, dtype="float32")
    sd.wait()
    print("  ✓  Done.")
    return frames.squeeze()


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file and resample to target_sr."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def mix(voice: np.ndarray, background: np.ndarray, bg_gain: float = 0.4) -> np.ndarray:
    """Mix voice with background at bg_gain level, trimmed to voice length."""
    n = len(voice)
    if len(background) < n:
        reps = (n // len(background)) + 1
        background = np.tile(background, reps)
    bg = background[:n] * bg_gain
    mixed = voice + bg
    # Normalise to avoid clipping
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed /= peak
    return mixed


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(rms + 1e-9)


def separate(wav_bytes: bytes, description: str, server: str) -> tuple[bytes, bytes]:
    resp = requests.post(
        f"{server}/v1/audio/separations",
        files={"file": ("recording.wav", wav_bytes, "audio/wav")},
        data={"description": description},
        timeout=180,
    )
    if resp.status_code != 200:
        print(f"  ✗  Server {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    target = base64.b64decode(data["target_url"].split(",")[1])
    residual = base64.b64decode(data["residual_url"].split(",")[1])
    return target, residual


def play_and_stats(wav_bytes: bytes, label: str):
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    db = rms_db(audio)
    print(f"  ▶  {label}  ({len(audio)/sr:.1f}s  RMS {db:.1f} dBFS)")
    sd.play(audio, sr)
    sd.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--description", type=str, default="speech")
    ap.add_argument("--server", type=str, default="http://localhost:8001")
    ap.add_argument("--mix", type=str, default=None,
                    help="Path to background audio file to mix in before sending")
    ap.add_argument("--bg-gain", type=float, default=0.4,
                    help="Background mix level 0-1 (default 0.4)")
    ap.add_argument("--save", type=str, default=None,
                    help="Save output as <path>_target.wav / <path>_residual.wav")
    args = ap.parse_args()

    try:
        health = requests.get(f"{args.server}/health", timeout=5).json()
    except Exception as e:
        print(f"  ✗  Can't reach {args.server}: {e}", file=sys.stderr)
        sys.exit(1)

    if not health.get("sts_model"):
        print("  ✗  No STS model loaded on server.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Server  : {args.server}")
    print(f"  Model   : {health['sts_model']}")
    print(f"  Extract : \"{args.description}\"")
    if args.mix:
        print(f"  BG mix  : {args.mix} (gain {args.bg_gain})")
    print()
    print("  SAM-Audio separates audio by description.")
    print("  For best results: play music or background noise from speakers while recording.")
    print()

    bg = None
    if args.mix:
        bg = load_audio(args.mix, MODEL_SR)

    while True:
        input("  [ Enter ] to record    Ctrl-C to quit\n")
        voice = record(args.duration)

        if bg is not None:
            print(f"  Mixing background at gain {args.bg_gain} …")
            signal = mix(voice, bg, args.bg_gain)
        else:
            signal = voice

        input_db = rms_db(signal)
        print(f"  Input RMS: {input_db:.1f} dBFS")

        wav_bytes = to_wav_bytes(signal, MODEL_SR)
        print(f"  Sending {len(wav_bytes)//1024}KB to server …")

        t0 = time.time()
        target, residual = separate(wav_bytes, args.description, args.server)
        print(f"  ✓  Separated in {time.time()-t0:.1f}s\n")

        play_and_stats(target,   f"TARGET   — should contain: \"{args.description}\"")
        play_and_stats(residual, f"RESIDUAL — everything else")

        if args.save:
            for data, suffix in [(target, "_target.wav"), (residual, "_residual.wav")]:
                path = args.save + suffix
                with open(path, "wb") as f:
                    f.write(data)
                print(f"  💾  {path}")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Bye!")
