"""Download GGUF weights from the Hugging Face Hub into a local models directory.

Files are stored as ``<models_dir>/<org>__<repo>/<file>.gguf`` so two repos can ship a
file with the same name. The directory lives outside the git repo (GGUF files are GBs).

CLI::

    python -m llmbench.models.hub --all                       # every enabled GGUF variant
    python -m llmbench.models.hub --model qwen2.5-1.5b --quant gguf_q4_k_m
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llmbench.config import (
    DEFAULT_BENCH_CONFIG,
    DEFAULT_MODELS_CONFIG,
    ModelsConfig,
    Variant,
    load_bench_config,
    load_models_config,
)
from llmbench.log import get_logger

log = get_logger(__name__)


def gguf_relpath(variant: Variant) -> Path:
    """Path of a GGUF file relative to the models directory."""
    if not variant.is_gguf or variant.repo_id is None or variant.file is None:
        raise ValueError(f"variant '{variant.quant}' is not a GGUF variant")
    return Path(variant.repo_id.replace("/", "__")) / variant.file


def ensure_gguf(variant: Variant, models_dir: Path) -> Path:
    """Return the local path of the GGUF file, downloading it if missing."""
    rel = gguf_relpath(variant)
    path = models_dir / rel
    if path.is_file():
        log.info("model already downloaded", extra={"path": str(path)})
        return path

    from huggingface_hub import hf_hub_download  # heavy import, only when downloading

    assert variant.repo_id is not None and variant.file is not None
    log.info("downloading", extra={"repo": variant.repo_id, "file": variant.file})
    downloaded = hf_hub_download(
        repo_id=variant.repo_id,
        filename=variant.file,
        local_dir=models_dir / rel.parent,
        token=os.getenv("HF_TOKEN") or None,
    )
    return Path(downloaded)


def _select(cfg: ModelsConfig, model: str | None, quant: str | None) -> list[tuple[str, Variant]]:
    if model is None:
        return [(m, v) for m, v in cfg.iter_variants(gpu_available=True) if v.is_gguf]
    spec = cfg.get(model)
    variants = [spec.variant(quant)] if quant else [v for v in spec.variants if v.enabled]
    return [(model, v) for v in variants if v.is_gguf]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download GGUF weights.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every enabled GGUF variant")
    group.add_argument("--model", help="model id from configs/models.yaml")
    parser.add_argument("--quant", help="quant of --model (default: all its GGUF variants)")
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--bench-config", default=DEFAULT_BENCH_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_models_config(args.models_config)
    models_dir = load_bench_config(args.bench_config).server.models_path
    for model_id, variant in _select(cfg, args.model, args.quant):
        path = ensure_gguf(variant, models_dir)
        size_gb = path.stat().st_size / 1024**3
        print(f"{model_id:<16} {variant.quant.value:<12} {size_gb:5.2f} GB  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
