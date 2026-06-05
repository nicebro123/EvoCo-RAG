"""Model weight and LoRA adapter path helpers.

The project keeps immutable base model weights outside this repo, while EvoCo-RAG
checkpoints are saved under the configured output directory:

    small base model: ../rag_assets/base_models/reranker/bge-reranker-v2-m3
    large base model: ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
    small LoRA rounds: ../rag_assets/checkpoints/evoco_popqa/small/round_000
    large LoRA rounds: ../rag_assets/checkpoints/evoco_popqa/large/round_000

These helpers prevent training/eval scripts from accidentally loading a
checkpoint root such as ../rag_assets/checkpoints/.../small as if it were a
PEFT adapter.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Optional


ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_MODEL_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
ROUND_RE = re.compile(r"^round_(\d+)$")


def is_lora_adapter_dir(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, ADAPTER_CONFIG)):
        return False
    return any(os.path.exists(os.path.join(path, name)) for name in ADAPTER_MODEL_FILES)


def adapter_rounds(root: Optional[str]) -> list[tuple[int, str]]:
    """Return complete round_* adapters under root as (round_id, path)."""
    if not root or not os.path.isdir(root):
        return []
    candidates: list[tuple[int, str]] = []
    for name in os.listdir(root):
        match = ROUND_RE.match(name)
        if not match:
            continue
        path = os.path.join(root, name)
        if is_lora_adapter_dir(path):
            candidates.append((int(match.group(1)), path))
    return sorted(candidates, key=lambda x: x[0])


def latest_round_adapter(root: Optional[str]) -> Optional[str]:
    """Return the newest round_* adapter under root, or None if none exists."""
    candidates = adapter_rounds(root)
    if not candidates:
        return None
    return candidates[-1][1]


def latest_checkpoint_round(root: Optional[str]) -> Optional[int]:
    candidates = adapter_rounds(root)
    return candidates[-1][0] if candidates else None


def resolve_adapter_for_loading(path_or_root: Optional[str]) -> Optional[str]:
    """Resolve either an adapter dir or a checkpoint root to a loadable adapter.

    Returns None when the path does not exist, is empty, or contains no complete
    PEFT adapter. This is intentional: callers can then initialize a fresh LoRA.
    """
    if not path_or_root:
        return None
    if is_lora_adapter_dir(path_or_root):
        return path_or_root
    return latest_round_adapter(path_or_root)


def checkpoint_round_dir(root: str, round_id: int) -> str:
    return os.path.join(root, f"round_{round_id:03d}")


def prepare_weight_layout(config, create: bool = True) -> dict:
    """Return and optionally create all weight/checkpoint directories."""
    layout = {
        "small_base_path": config.models.small_base_path,
        "large_base_path": config.models.large_base_path,
        "small_checkpoint_root": config.models.small_lora_dir,
        "large_checkpoint_root": config.models.large_lora_dir,
        "small_latest_adapter": resolve_adapter_for_loading(config.models.small_lora_dir),
        "large_latest_adapter": resolve_adapter_for_loading(config.models.large_lora_dir),
        "small_latest_round": latest_checkpoint_round(config.models.small_lora_dir),
        "large_latest_round": latest_checkpoint_round(config.models.large_lora_dir),
        "legacy_small_adapter": "../rag_assets/adapters/reranker-CoRAG",
        "legacy_large_adapter": "../rag_assets/adapters/generator-CoRAG",
    }
    if create:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.models.small_lora_dir, exist_ok=True)
        os.makedirs(config.models.large_lora_dir, exist_ok=True)
        for sub in ("replay", "contracts", "audits", "metrics"):
            os.makedirs(os.path.join(config.output_dir, sub), exist_ok=True)
    return layout


def write_weight_manifest(config, output_dir: Optional[str] = None) -> str:
    """Persist the authoritative weight layout for a run."""
    out_dir = output_dir or config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    layout = prepare_weight_layout(config, create=True)
    manifest = {
        "project": {
            "name": config.name,
            "seed": config.seed,
            "output_dir": config.output_dir,
        },
        "weights": layout,
        "models_config": asdict(config.models),
        "runtime_config": asdict(config.runtime),
    }
    path = os.path.join(out_dir, "weights_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
