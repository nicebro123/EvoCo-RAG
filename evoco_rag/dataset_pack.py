"""Dataset-pack path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_DATA_ROOT = "../rag_assets/rag_data/evoco_dataset_pack"
REGISTRY_NAME = "dataset_registry.json"


def candidate_dataset_roots(path: Path) -> Iterable[Path]:
    """Yield plausible dataset-pack roots for common rag_assets layouts."""
    yield path
    yield path / "evoco_dataset_pack"
    yield path / "rag_data" / "evoco_dataset_pack"
    if path.name == "evoco_dataset_pack":
        yield path.parent / "rag_data" / "evoco_dataset_pack"


def resolve_dataset_pack_root(raw_path: str | Path) -> Path:
    base = Path(raw_path).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base

    checked: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidate_dataset_roots(base):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(resolved)
        if (resolved / REGISTRY_NAME).exists():
            return resolved

    checked_text = "\n  ".join(str(path / REGISTRY_NAME) for path in checked)
    raise FileNotFoundError(
        "dataset registry not found. Expected an extracted evoco_dataset_pack at one of:\n"
        f"  {checked_text}"
    )
