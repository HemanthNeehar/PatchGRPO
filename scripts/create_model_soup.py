"""Model Soup script for blending LoRA checkpoints.

Supports dual-checkpoint blending (e.g. Step 8 + Step 34 'best')
and tri-checkpoint blending (e.g. Step 4 + Step 8 + Step 34).
Reference: Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", ICML 2022.
"""

import argparse
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def blend_adapters(
    checkpoint_weights: list[tuple[Path, float]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    total_weight = sum(w for _, w in checkpoint_weights)
    print(f"[*] Blending {len(checkpoint_weights)} checkpoints with total weight sum = {total_weight:.4f}")

    # Verify all paths exist
    for p, w in checkpoint_weights:
        sf_path = p / "adapter_model.safetensors"
        if not sf_path.exists():
            raise FileNotFoundError(f"Missing adapter_model.safetensors in {p}")
        print(f"  - {p.name}: raw_weight = {w:.3f} (norm = {w/total_weight:.3f})")

    # Load base keys from the first checkpoint
    primary_dir, _ = checkpoint_weights[0]
    with safe_open(primary_dir / "adapter_model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}

    print(f"[*] Total tensors to interpolate: {len(keys)}")

    # Open all checkpoints
    readers = [
        (safe_open(p / "adapter_model.safetensors", framework="pt"), w / total_weight)
        for p, w in checkpoint_weights
    ]

    blended_tensors = {}
    for key in keys:
        accum = None
        for reader, norm_w in readers:
            t = reader.get_tensor(key).to(torch.float32)
            if accum is None:
                accum = t * norm_w
            else:
                accum += t * norm_w
        blended_tensors[key] = accum.contiguous()

    # Save blended safetensors
    out_sf = output_dir / "adapter_model.safetensors"
    save_file(blended_tensors, str(out_sf), metadata=metadata)
    print(f"[OK] Saved blended safetensors: {out_sf} ({out_sf.stat().st_size / 1024 / 1024:.1f} MB)")

    # Copy adapter_config.json and README.md from primary
    for fname in ["adapter_config.json", "README.md"]:
        src = primary_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"[OK] Copied {fname} to {output_dir}")

    # Remove any unwanted optimizer or trainer state files in output_dir
    for junk in output_dir.glob("optimizer*"):
        junk.unlink()
    for junk in output_dir.glob("trainer_state*"):
        junk.unlink()


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Souping")
    parser.add_argument("--step4_dir", type=str, default="main_lora_online_step4-20261107T113313Z-1-001/main_lora_online_step4")
    parser.add_argument("--step8_dir", type=str, default="main_lora_online_step8-20261107T113311Z-1-001/main_lora_online_step8")
    parser.add_argument("--best_dir", type=str, default="main_lora_best-20261106T155831Z-1-001/main_lora_best")
    parser.add_argument("--mode", type=str, choices=["dual", "tri"], default="dual",
                        help="Mode: 'dual' (Step 8 + Best) or 'tri' (Step 4 + Step 8 + Best)")
    parser.add_argument("--step8_weight", type=float, default=0.70)
    parser.add_argument("--best_weight", type=float, default=0.30)
    parser.add_argument("--step4_weight", type=float, default=0.20)
    parser.add_argument("--output_dir", type=str, default="gemma-4-developer-agent/submission/adapters/main_lora")
    args = parser.parse_args()

    if args.mode == "tri":
        checkpoints = [
            (Path(args.step8_dir), 0.50),
            (Path(args.step4_dir), 0.25),
            (Path(args.best_dir), 0.25),
        ]
    else:
        checkpoints = [
            (Path(args.step8_dir), args.step8_weight),
            (Path(args.best_dir), args.best_weight),
        ]
    blend_adapters(checkpoints, Path(args.output_dir))


if __name__ == "__main__":
    main()
