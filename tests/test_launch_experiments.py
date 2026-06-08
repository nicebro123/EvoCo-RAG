import subprocess
import sys
from pathlib import Path

import yaml


def _write_base_config(path: Path) -> None:
    path.write_text(
        """
project:
  name: base
  seed: 1
  output_dir: ../rag_assets/outputs/base
data:
  train_path: ../rag_assets/evoco_dataset_pack/datasets/popqa_standard/data_v33/Pop/train_labels_list.json
  test_path: ../rag_assets/evoco_dataset_pack/datasets/popqa_standard/data/Pop/test.json
  dataset_name: PopQAStandard
  debug_size: 16
models:
  small_base_path: ../rag_assets/base_models/reranker/bge-reranker-v2-m3
  large_base_path: ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
  small_lora_dir: ../rag_assets/checkpoints/base/small
  large_lora_dir: ../rag_assets/checkpoints/base/large
contract:
  top_k: 3
  max_selected_docs: 3
runtime:
  audit_batch_size: 1
training:
  num_rounds: 1
  batch_size: 1
  large_batch_size: 1
""",
        encoding="utf-8",
    )


def test_launch_experiments_dry_run_materializes_configs(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "unit_study",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0,1", "seed": 7},
                "experiments": [
                    {"name": "base_run"},
                    {
                        "name": "top5",
                        "overrides": {
                            "contract.top_k": 5,
                            "contract.max_selected_docs": 5,
                            "runtime.audit_batch_size": 2,
                        },
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/launch_experiments.py", "--spec", str(spec)],
        check=True,
        text=True,
        capture_output=True,
    )

    study_dir = output_root / "unit_study"
    first = yaml.safe_load((study_dir / "00_base_run_s7_g0_1" / "run_config.yaml").read_text(encoding="utf-8"))
    second = yaml.safe_load((study_dir / "01_top5_s7_g0_1" / "run_config.yaml").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((study_dir / "launch_manifest.yaml").read_text(encoding="utf-8"))

    assert "Dry run complete" in result.stdout
    assert first["project"]["name"] == "00_base_run_s7_g0_1"
    assert first["project"]["seed"] == 7
    assert first["project"]["output_dir"] == str(study_dir / "00_base_run_s7_g0_1")
    assert first["models"]["small_lora_dir"] == str(checkpoint_root / "unit_study" / "00_base_run_s7_g0_1" / "small")
    assert second["contract"]["top_k"] == 5
    assert second["runtime"]["audit_batch_size"] == 2
    assert len(manifest["runs"]) == 2
    assert (study_dir / "run_gpu0_1.sh").exists()
    assert (study_dir / "launch_tmux.sh").exists()
