import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evoco_rag.config import EvoCoConfig
from scripts.launch_experiments import apply_overrides


def _write_base_config(path: Path) -> None:
    path.write_text(
        """
project:
  name: base
  seed: 1
  output_dir: ../rag_assets/outputs/base
data:
  train_path: ../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data_v33/Pop/train_labels_list.json
  test_path: ../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data/Pop/test.json
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


def test_stale_override_is_rejected_before_launch():
    config = apply_overrides({}, {"training.policy_num_generations": 2})
    with pytest.raises(ValueError, match="unknown config keys in training"):
        EvoCoConfig.from_dict(config)


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
    gpu_script = (study_dir / "run_gpu0_1.sh").read_text(encoding="utf-8")
    tmux_script = (study_dir / "launch_tmux.sh").read_text(encoding="utf-8")

    assert "Dry run complete" in result.stdout
    assert "scripts/eval_evoco.py" in result.stdout
    assert first["project"]["name"] == "00_base_run_s7_g0_1"
    assert first["project"]["seed"] == 7
    assert first["project"]["output_dir"] == str(study_dir / "00_base_run_s7_g0_1")
    assert first["models"]["small_lora_dir"] == str(checkpoint_root / "unit_study" / "00_base_run_s7_g0_1" / "small")
    assert second["contract"]["top_k"] == 5
    assert second["runtime"]["audit_batch_size"] == 2
    assert len(manifest["runs"]) == 2
    assert manifest["evaluation_protocol_version"] == 2
    assert manifest["runs"][0]["eval_after_train"] is True
    assert manifest["runs"][0]["eval_log_path"].endswith("eval.log")
    assert manifest["runs"][0]["train_marker_path"].endswith("metrics/round_000.json")
    assert manifest["runs"][0]["completion_marker_path"].endswith("metrics/test_eval.json")
    assert "scripts/eval_evoco.py" in manifest["runs"][0]["eval_command"]
    assert (study_dir / "run_gpu0_1.sh").exists()
    assert "metrics/test_eval.json" in gpu_script
    assert "eval.log" in gpu_script
    assert "train complete marker found" in gpu_script
    assert "final-round test metrics found; skip duplicate eval" in gpu_script
    assert "tmux has-session" in tmux_script
    assert (study_dir / "launch_tmux.sh").exists()


def test_launch_experiments_respects_env_gpu_override(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "gpu_override",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [{"name": "base_run"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["EVOCO_GPUS"] = "2,3"

    subprocess.run(
        [sys.executable, "scripts/launch_experiments.py", "--spec", str(spec)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    study_dir = output_root / "gpu_override"
    manifest = yaml.safe_load((study_dir / "launch_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["runs"][0]["gpu"] == "2,3"
    assert (study_dir / "00_base_run_s7_g2_3" / "run_config.yaml").exists()
    assert (study_dir / "run_gpu2_3.sh").exists()


def test_launch_experiments_round_robins_env_gpu_pairs(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "gpu_pairs",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [
                    {"name": "run0"},
                    {"name": "run1"},
                    {"name": "run2"},
                    {"name": "run3"},
                    {"name": "run4"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["EVOCO_GPU_PAIRS"] = "0,1;2,3;4,5;6,7"

    subprocess.run(
        [sys.executable, "scripts/launch_experiments.py", "--spec", str(spec)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    study_dir = output_root / "gpu_pairs"
    manifest = yaml.safe_load((study_dir / "launch_manifest.yaml").read_text(encoding="utf-8"))
    assert [run["gpu"] for run in manifest["runs"]] == ["0,1", "2,3", "4,5", "6,7", "0,1"]
    assert (study_dir / "run_gpu0_1.sh").exists()
    assert (study_dir / "run_gpu2_3.sh").exists()
    assert (study_dir / "run_gpu4_5.sh").exists()
    assert (study_dir / "run_gpu6_7.sh").exists()


def test_launch_experiments_launch_tmux_invokes_generated_launcher(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    fake_bin = tmp_path / "bin"
    bash_calls = tmp_path / "bash_calls.txt"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        f"""#!/usr/bin/env python3
import sys
from pathlib import Path

Path({str(bash_calls)!r}).write_text("\\n".join(sys.argv[1:]), encoding="utf-8")
print("fake bash invoked " + " ".join(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "tmux_study",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [{"name": "one"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_experiments.py",
            "--spec",
            str(spec),
            "--launch-tmux",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    study_dir = output_root / "tmux_study"
    assert "Started tmux queues with: bash" in result.stdout
    assert "fake bash invoked" in result.stdout
    assert bash_calls.read_text(encoding="utf-8") == str(study_dir / "launch_tmux.sh")
    assert (study_dir / "run_gpu0.sh").exists()
    assert (study_dir / "launch_tmux.sh").exists()


def test_launch_tmux_sh_dry_run_forwards_to_launcher(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "bash_entry",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [{"name": "one"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/launch_tmux.sh",
            "--spec",
            str(spec),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    study_dir = output_root / "bash_entry"
    assert "mode: dry run" in result.stdout
    assert "Dry run complete" in result.stdout
    assert (study_dir / "00_one_s7_g0" / "run_config.yaml").exists()
    assert (study_dir / "launch_tmux.sh").exists()


def test_launch_experiments_launch_runs_eval_after_train(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    train_script = tmp_path / "fake_train.py"
    eval_script = tmp_path / "fake_eval.py"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    train_script.write_text(
        """
import argparse
import json
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
out = Path(cfg['project']['output_dir']) / 'metrics'
out.mkdir(parents=True, exist_ok=True)
(out / 'round_000.json').write_text(json.dumps({'trained': True}), encoding='utf-8')
print('fake train done')
""",
        encoding="utf-8",
    )
    eval_script.write_text(
        """
import argparse
import json
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
out = Path(cfg['project']['output_dir']) / 'metrics'
out.mkdir(parents=True, exist_ok=True)
(out / 'test_eval.json').write_text(json.dumps({'accuracy': 1.0}), encoding='utf-8')
print('fake eval done')
""",
        encoding="utf-8",
    )
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "launch_eval",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "python": sys.executable,
                "train_script": str(train_script),
                "eval_script": str(eval_script),
                "experiments": [{"name": "one"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_experiments.py",
            "--spec",
            str(spec),
            "--launch",
            "--no-gpu-scripts",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    run_dir = output_root / "launch_eval" / "00_one"
    assert "fake train done" in result.stdout
    assert "fake eval done" in result.stdout
    assert (run_dir / "metrics" / "round_000.json").exists()
    assert (run_dir / "metrics" / "test_eval.json").exists()
    assert "fake train done" in (run_dir / "train.log").read_text(encoding="utf-8")
    assert "fake eval done" in (run_dir / "eval.log").read_text(encoding="utf-8")


def test_launch_experiments_launch_eval_only_when_training_marker_exists(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    train_script = tmp_path / "fake_train.py"
    eval_script = tmp_path / "fake_eval.py"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    train_script.write_text(
        """
raise SystemExit('fake train should not run')
""",
        encoding="utf-8",
    )
    eval_script.write_text(
        """
import argparse
import json
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
out = Path(cfg['project']['output_dir']) / 'metrics'
out.mkdir(parents=True, exist_ok=True)
(out / 'test_eval.json').write_text(json.dumps({'accuracy': 1.0}), encoding='utf-8')
print('fake eval done')
""",
        encoding="utf-8",
    )
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "eval_only",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "python": sys.executable,
                "train_script": str(train_script),
                "eval_script": str(eval_script),
                "experiments": [{"name": "one"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    train_marker = output_root / "eval_only" / "00_one" / "metrics" / "round_000.json"
    train_marker.parent.mkdir(parents=True, exist_ok=True)
    train_marker.write_text('{"trained": true}', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_experiments.py",
            "--spec",
            str(spec),
            "--launch",
            "--no-gpu-scripts",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    run_dir = output_root / "eval_only" / "00_one"
    assert "SKIP training; found train marker" in result.stdout
    assert "fake eval done" in result.stdout
    assert "fake train should not run" not in result.stdout
    assert (run_dir / "metrics" / "test_eval.json").exists()
    assert "fake eval done" in (run_dir / "eval.log").read_text(encoding="utf-8")


def test_launch_all_experiments_dry_run_forwards_specs(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "all_entry",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [{"name": "one"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/launch_all_experiments.sh",
            "--skip-verify",
            "--no-generate-configs",
            "--dry-run",
            "--spec",
            str(spec),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    study_dir = output_root / "all_entry"
    assert "mode: dry run" in result.stdout
    assert "Dry run complete for 1 spec(s)." in result.stdout
    assert (study_dir / "00_one_s7_g0" / "run_config.yaml").exists()
    assert (study_dir / "run_gpu0.sh").exists()


def test_launch_all_experiments_launch_uses_master_tmux_queue(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    fake_bin = tmp_path / "bin"
    tmux_calls = tmp_path / "tmux_calls.txt"
    master_dir = tmp_path / "master"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        f"""#!/usr/bin/env bash
echo "$@" >> {str(tmux_calls)!r}
if [[ "$1" == "has-session" ]]; then
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "all_launch",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [{"name": "one"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "EVOCO_MASTER_OUTPUT_DIR": str(master_dir),
        "EVOCO_TMUX_SESSION": "unit_all",
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/launch_all_experiments.sh",
            "--skip-verify",
            "--no-generate-configs",
            "--spec",
            str(spec),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    master_script = master_dir / "run_all_gpu_queues.sh"
    calls = tmux_calls.read_text(encoding="utf-8")
    assert "Started master tmux queue: unit_all" in result.stdout
    assert "has-session -t unit_all" in calls
    assert "new -d -s unit_all" in calls
    assert master_script.exists()
    assert str(output_root / "all_launch" / "run_gpu0.sh") in master_script.read_text(encoding="utf-8")


def test_launch_all_experiments_gpu_pairs_start_worker_queues(tmp_path):
    base_config = tmp_path / "base.yaml"
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "checkpoints"
    spec = tmp_path / "spec.yaml"
    fake_bin = tmp_path / "bin"
    tmux_calls = tmp_path / "tmux_calls.txt"
    master_dir = tmp_path / "master"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        f"""#!/usr/bin/env bash
echo "$@" >> {str(tmux_calls)!r}
if [[ "$1" == "has-session" ]]; then
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    _write_base_config(base_config)
    spec.write_text(
        yaml.safe_dump(
            {
                "study_name": "all_pairs",
                "output_root": str(output_root),
                "checkpoint_root": str(checkpoint_root),
                "base_config": str(base_config),
                "defaults": {"gpu": "0", "seed": 7},
                "experiments": [
                    {"name": "one"},
                    {"name": "two"},
                    {"name": "three"},
                    {"name": "four"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "EVOCO_MASTER_OUTPUT_DIR": str(master_dir),
        "EVOCO_TMUX_SESSION": "unit_all",
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/launch_all_experiments.sh",
            "--skip-verify",
            "--no-generate-configs",
            "--gpu-pairs",
            "0,1;2,3",
            "--spec",
            str(spec),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    calls = tmux_calls.read_text(encoding="utf-8")
    worker0 = master_dir / "workers" / "run_gpu0_1_queue.sh"
    worker1 = master_dir / "workers" / "run_gpu2_3_queue.sh"
    assert "Started 2 GPU-pair worker queue(s)." in result.stdout
    assert "new -d -s unit_all_g0_1" in calls
    assert "new -d -s unit_all_g2_3" in calls
    assert worker0.exists()
    assert worker1.exists()
    assert str(output_root / "all_pairs" / "run_gpu0_1.sh") in worker0.read_text(encoding="utf-8")
    assert str(output_root / "all_pairs" / "run_gpu2_3.sh") in worker1.read_text(encoding="utf-8")


def test_official_popqa_experiment_specs_cover_main_studies():
    specs = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in Path("configs/experiments").glob("popqa_*_2gpu.yaml")
    }
    assert {
        "popqa_sweep_full_2gpu.yaml",
        "popqa_hparam_full_2gpu.yaml",
        "popqa_full_sweep_2gpu.yaml",
        "popqa_ablation_full_2gpu.yaml",
    }.issubset(specs)

    sweep_names = {item["name"] for item in specs["popqa_sweep_full_2gpu.yaml"]["experiments"]}
    assert {"cost_top3", "precision_top5", "precision_top8", "answer_only_reward", "no_audit"} == sweep_names

    ablation_names = {item["name"] for item in specs["popqa_ablation_full_2gpu.yaml"]["experiments"]}
    assert {
        "evoco_full",
        "large_sft_only",
        "answer_only_reward",
        "no_audit",
        "small_only",
        "large_only",
        "baseline_no_audit_answer_only",
    } == ablation_names

    hparam_names = {item["name"] for item in specs["popqa_hparam_full_2gpu.yaml"]["experiments"]}
    assert {"top3_audit1_base", "top5_audit3", "top8_audit5"}.issubset(hparam_names)
    assert {"high_conf_065_top5", "high_conf_085_top5"}.issubset(hparam_names)
    assert "grpo_gen4_top5" in hparam_names

    launch_all = Path("scripts/launch_all_experiments.sh").read_text(encoding="utf-8")
    assert "configs/experiments/popqa_llama8b_full_sweep_2gpu.yaml" in launch_all
    retired_specs = {
        "popqa_sweep_full_2gpu.yaml",
        "popqa_hparam_full_2gpu.yaml",
        "multidataset_full_2gpu.yaml",
        "popqa_full_sweep_2gpu.yaml",
        "popqa_ablation_full_2gpu.yaml",
    }
    for spec_name in retired_specs:
        assert spec_name not in launch_all
    assert "configs/experiments/popqa_fast_sweep_2gpu.yaml" not in launch_all
    assert "configs/experiments/popqa_hparam_fast_2gpu.yaml" not in launch_all
    assert "configs/experiments/multidataset_fast_2gpu.yaml" not in launch_all
