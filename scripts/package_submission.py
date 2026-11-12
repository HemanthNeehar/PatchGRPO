#!/usr/bin/env python3
"""Validate and package `gemma-4-developer-agent/submission/` into `submission.zip`.

Mirrors the structural rules in HARNESS_README.md (sections 2-3) because `adk-submission` is not
installed on the local machine. Run from the repository root:

    python scripts/package_submission.py                 # validate + build submission.zip
    python scripts/package_submission.py --check-only    # validate only
    python scripts/package_submission.py --allow-placeholder-adapters   # dry-run packaging only

Packaging notes:
  * `agent.yaml` sits at the ROOT of the zip (archive is built from inside `submission/`).
  * Entries use forward slashes (Windows `Compress-Archive` may write backslashes).
  * The script refuses to package placeholder adapters unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = REPO_ROOT / "gemma-4-developer-agent" / "submission"
DEFAULT_ZIP = REPO_ROOT / "gemma-4-developer-agent" / "submission.zip"

COMPETITION_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
ALLOWED_ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
MAX_TOTAL_BYTES = 3 * 1024 ** 3
MAX_FILES = 10_000
MAX_ADAPTERS = 8
MAX_LORA_RANK = 128
MAX_TOKENS = 32_768


class _Loader(yaml.SafeLoader):
    """YAML loader that resolves `!include` inside the submission root only."""


def _make_loader(root: Path, problems: list[str]):
    def _include(loader: _Loader, node: yaml.Node):
        rel = loader.construct_scalar(node)
        target = (Path(loader.name).parent / rel).resolve()
        if root not in target.parents:
            problems.append(f"!include escapes submission root: {rel}")
            return ""
        if not target.exists():
            problems.append(f"!include target missing: {rel}")
            return ""
        if target.suffix in (".md", ".txt"):
            return target.read_text(encoding="utf-8")
        return _load_yaml(target, root, problems)

    class Loader(_Loader):
        pass

    Loader.add_constructor("!include", _include)
    return Loader


def _load_yaml(path: Path, root: Path, problems: list[str]):
    loader_cls = _make_loader(root, problems)
    with open(path, encoding="utf-8") as fh:
        loader = loader_cls(fh)
        loader.name = str(path)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


def _safetensors_summary(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    keys = [k for k in header if k != "__metadata__"]
    params = sum(int(math.prod(header[k]["shape"])) for k in keys)
    return len(keys), params


def validate_submission(root: Path, allow_placeholder: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Returns `(problems, warnings, notes)`."""
    problems: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not root.is_dir():
        return [f"submission directory not found: {root}"], warnings, notes

    files = [p for p in root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    notes.append(f"{len(files)} files, {total / 1024 ** 2:.2f} MiB (limit 3 GiB)")
    if total >= MAX_TOTAL_BYTES:
        problems.append("unpacked size >= 3 GiB")
    if len(files) > MAX_FILES:
        problems.append(f"more than {MAX_FILES} files")
    for p in root.rglob("*"):
        if p.is_symlink():
            problems.append(f"symlink not allowed: {p.relative_to(root)}")
    for p in files:
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            problems.append(f"disallowed file extension: {p.relative_to(root)}")
        if "__pycache__" in p.parts:
            problems.append(f"remove cache artifact: {p.relative_to(root)}")
    for p in root.glob("*.py"):
        problems.append(f"Python file in submission root is not allowed: {p.name}")

    roots = [n for n in ROOT_CONFIG_NAMES if (root / n).exists()]
    if len(roots) != 1:
        problems.append(f"exactly one root config required, found: {roots}")
        return problems, warnings, notes

    models: set[str] = set()
    adapters_used: set[str] = set()

    def walk(cfg_path: Path) -> None:
        cfg = _load_yaml(cfg_path, root, problems) or {}
        if cfg.get("model"):
            models.add(str(cfg["model"]).split("/")[-1])
        if cfg.get("adapter"):
            adapters_used.add(cfg["adapter"])
        gen = cfg.get("generate_content_config") or {}
        if gen.get("max_output_tokens", 1) > MAX_TOKENS:
            problems.append(f"{cfg_path.name}: max_output_tokens > {MAX_TOKENS}")
        if (gen.get("thinking_config") or {}).get("thinking_budget", 0) > MAX_TOKENS:
            problems.append(f"{cfg_path.name}: thinking_budget > {MAX_TOKENS}")
        for forbidden in ("tools", "system_instruction", "http_options", "safety_settings", "response_schema"):
            if forbidden in gen:
                problems.append(f"{cfg_path.name}: forbidden generate_content_config field '{forbidden}'")
        for tool in cfg.get("tools", []) or []:
            if isinstance(tool, dict) and "agent_tool" in tool:
                walk(root / tool["agent_tool"]["config_path"])
        for sub in cfg.get("sub_agents", []) or []:
            walk(root / sub["config_path"])

    walk(root / roots[0])
    notes.append(f"declared models: {sorted(models)}; adapters referenced: {sorted(adapters_used)}")
    if models != {COMPETITION_MODEL}:
        problems.append(f"all agents must declare exactly '{COMPETITION_MODEL}', found {sorted(models)}")

    adapters_dir = root / "adapters"
    on_disk = sorted(d.name for d in adapters_dir.iterdir() if d.is_dir()) if adapters_dir.exists() else []
    if len(on_disk) > MAX_ADAPTERS:
        problems.append(f"more than {MAX_ADAPTERS} adapters")
    for name in sorted(adapters_used):
        if name not in on_disk:
            problems.append(f"adapter '{name}' is referenced in YAML but missing from adapters/")
    for name in on_disk:
        if name not in adapters_used:
            warnings.append(f"adapter '{name}' exists but no agent references it")

    seen_hashes: dict[str, str] = {}
    for name in on_disk:
        d = adapters_dir / name
        extra = {x.name for x in d.iterdir()} - ALLOWED_ADAPTER_FILES
        if extra:
            problems.append(f"adapter '{name}' has unexpected files: {sorted(extra)}")
        cfg_file, w_file = d / "adapter_config.json", d / "adapter_model.safetensors"
        if not cfg_file.exists() or not w_file.exists():
            problems.append(f"adapter '{name}' needs adapter_config.json and adapter_model.safetensors")
            continue
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        rank = int(cfg.get("r", 0))
        if rank > MAX_LORA_RANK:
            problems.append(f"adapter '{name}': rank {rank} > {MAX_LORA_RANK}")
        if cfg.get("peft_type") != "LORA":
            problems.append(f"adapter '{name}': peft_type must be LORA")
        n_tensors, n_params = _safetensors_summary(w_file)
        notes.append(
            f"adapter '{name}': r={rank} alpha={cfg.get('lora_alpha')} targets={cfg.get('target_modules')} "
            f"layers_to_transform={cfg.get('layers_to_transform')} tensors={n_tensors} params={n_params:,}"
        )
        digest = w_file.read_bytes()[:1 << 20]
        key = str(hash(digest)) + str(w_file.stat().st_size)
        if key in seen_hashes:
            warnings.append(f"adapter '{name}' is byte-identical to '{seen_hashes[key]}'")
        seen_hashes[key] = name

        placeholder = bool(cfg.get("layers_to_transform")) or rank < 8 or n_params < 5_000_000
        if placeholder:
            msg = (
                f"adapter '{name}' looks like a PLACEHOLDER (r={rank}, layers_to_transform="
                f"{cfg.get('layers_to_transform')}, params={n_params:,}); train and export a real adapter"
            )
            (warnings if allow_placeholder else problems).append(msg)
    return problems, warnings, notes


def build_zip(root: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, arcname=path.relative_to(root).as_posix())
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert any(n in ROOT_CONFIG_NAMES for n in names), "root config must be at the zip root"
    assert not any("\\" in n or n.startswith(("/", "submission/")) for n in names), "bad zip entry paths"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-placeholder-adapters", action="store_true")
    args = parser.parse_args()

    problems, warnings, notes = validate_submission(args.submission_dir.resolve(), args.allow_placeholder_adapters)
    for n in notes:
        print(f"[note]    {n}")
    for w in warnings:
        print(f"[warning] {w}")
    for p in problems:
        print(f"[ERROR]   {p}")
    if problems:
        print(f"\nValidation FAILED ({len(problems)} problem(s)). Nothing was packaged.")
        return 1
    print("\nValidation passed.")
    if args.check_only:
        return 0
    build_zip(args.submission_dir.resolve(), args.output.resolve())
    size_mb = args.output.stat().st_size / 1024 ** 2
    print(f"Wrote {args.output} ({size_mb:.2f} MiB) with agent.yaml at the archive root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
