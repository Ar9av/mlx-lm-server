#!/usr/bin/env python3
"""
Convert old mflux community models (pre-0.16) to mflux 0.18.0-compatible format.

Root cause: mflux 0.18.0 recognises its own saved models by a "quantization_level"
key in the safetensors metadata.  Community models lack this key, so mflux falls
through to the HF-format loading path which loads pre-quantized uint32 weights into
float32 Linear layers and then re-quantizes the garbage → garbled output.

This script:
  1. Loads every component directory (transformer, text_encoder, text_encoder_2, vae)
  2. Finds the 28 "broken" layers: bfloat16 weight stored inside a Quantized* slot
     (identified by a .weight in bfloat16 that has a sibling .scales or .biases key)
  3. Properly re-quantizes those layers with group_size=64, bits=4
  4. Re-saves each component with  {"quantization_level": "4"}  in the metadata

Usage:
    python tools/convert_mflux_model.py <hf_repo_or_local_path> <output_dir>

Examples:
    python tools/convert_mflux_model.py madroid/flux.1-schnell-mflux-4bit ~/flux-schnell-converted
    python tools/convert_mflux_model.py ~/.cache/huggingface/hub/models--madroid.../snapshots/... ~/flux-converted
"""

import shutil
import sys
from pathlib import Path

import mlx.core as mx


def resolve_model_path(src: str) -> Path:
    p = Path(src).expanduser()
    if p.exists():
        return p
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {src} from HuggingFace...")
        return Path(snapshot_download(src))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def load_component(src_dir: Path) -> dict:
    weights: dict[str, mx.array] = {}
    for f in sorted(src_dir.glob("*.safetensors")):
        w = mx.load(str(f))
        weights.update(w)
    return weights


def convert_component(src_dir: Path, dst_dir: Path, name: str) -> None:
    if not src_dir.exists():
        print(f"  [{name}] not found — skipping")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] loading {len(list(src_dir.glob('*.safetensors')))} shard(s)...")
    weights = load_component(src_dir)
    print(f"  [{name}] {len(weights)} tensors loaded")

    keys = set(weights)
    out: dict[str, mx.array] = {}
    skip: set[str] = set()
    n_fixed = 0

    for key in sorted(keys):
        if key in skip:
            continue

        val = weights[key]

        # Detect broken layer: bfloat16 weight whose base name also has .scales/.biases
        if key.endswith(".weight") and val.dtype == mx.bfloat16:
            base = key[: -len(".weight")]
            has_scales = f"{base}.scales" in keys
            has_biases = f"{base}.biases" in keys
            if has_scales or has_biases:
                # This layer was stored as plain float but wrapped as Quantized* by mflux.
                # Re-quantize it properly so it survives the stored_q load path.
                w_q, scales, biases = mx.quantize(val, group_size=64, bits=4)
                out[key] = w_q
                out[f"{base}.scales"] = scales
                out[f"{base}.biases"] = biases
                skip.add(f"{base}.scales")
                skip.add(f"{base}.biases")
                n_fixed += 1
                continue

        out[key] = val

    print(f"  [{name}] re-quantized {n_fixed} broken layer(s)")

    # Save as a single shard with the mflux metadata tag that triggers the
    # _try_load_mflux_format path in WeightLoader.
    out_file = dst_dir / "0.safetensors"
    mx.save_safetensors(
        str(out_file),
        out,
        metadata={"quantization_level": "4", "mflux_version": "0.18.0-converted"},
    )
    size_gb = out_file.stat().st_size / 1e9
    print(f"  [{name}] saved {out_file.name} ({size_gb:.2f} GB)")


def copy_tokenizers(src: Path, dst: Path) -> None:
    for name in ("tokenizer", "tokenizer_2"):
        s = src / name
        d = dst / name
        if s.exists():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
            print(f"  Copied {name}/")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src_arg, dst_arg = sys.argv[1], sys.argv[2]
    dst = Path(dst_arg).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(src_arg)
    print(f"\nSource : {model_path}")
    print(f"Output : {dst}\n")

    for comp in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        convert_component(model_path / comp, dst / comp, comp)

    print("\nCopying tokenizers...")
    copy_tokenizers(model_path, dst)

    print(f"\nConversion complete → {dst}")
    print(f"\nTest with:")
    print(f"  mflux-generate --model {dst} --prompt 'a red apple' --steps 4 --output /tmp/test.png")
    print(f"\nIn the image server:")
    print(f'  curl -X POST http://localhost:8002/v1/models/load \\')
    print(f'    -d \'{{"model": "flux-schnell", "model_path": "{dst}"}}\'')


if __name__ == "__main__":
    main()
