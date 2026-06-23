#!/usr/bin/env python3
"""Generate and optionally launch EvoCo-RAG experiment batches.

This mirrors the SpecFlow-style workflow:

* write one compact study spec under ``configs/experiments``;
* keep a base config stable;
* describe each run with dotted-key overrides;
* materialize an immutable ``run_config.yaml`` per run;
* write a ``launch_manifest.yaml`` plus per-GPU shell scripts;
* optionally run test-set evaluation immediately after training.

Dry-run is the default. Use ``--launch`` for foreground sequential execution or
``--launch-tmux`` to start the generated per-GPU queues in tmux.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.config import EvoCoConfig


DEFAULT_TRAIN_SCRIPT = "scripts/train_evoco.py"
DEFAULT_EVAL_SCRIPT = "scripts/eval_evoco.py"
EVALUATION_PROTOCOL_VERSION = 3


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def set_by_path(payload: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"invalid override key: {dotted_key!r}")
    current: MutableMapping[str, Any] = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        child = current[part]
        if not isinstance(child, MutableMapping):
            raise ValueError(f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def apply_overrides(base_config: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    for key, value in (overrides or {}).items():
        set_by_path(config, str(key), value)
    return config


def slug(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", "+"}:
            chars.append(char)
        elif char.isspace() or char in {",", "/"}:
            chars.append("_")
    return "".join(chars).strip("._-") or "run"


def quote_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def experiment_list(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("spec must contain a non-empty 'experiments' list")
    defaults = spec.get("defaults", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("spec 'defaults' must be a mapping")
    gpu_pairs = parse_gpu_pairs(os.environ.get("EVOCO_GPU_PAIRS"))
    gpu_override = os.environ.get("EVOCO_GPUS")
    output = []
    for index, item in enumerate(experiments):
        if not isinstance(item, dict):
            raise ValueError(f"experiments[{index}] must be a mapping")
        merged = dict(defaults)
        merged.update(item)
        if gpu_pairs:
            merged["gpu"] = gpu_pairs[index % len(gpu_pairs)]
        elif gpu_override:
            merged["gpu"] = gpu_override
        if "name" not in merged:
            raise ValueError(f"experiments[{index}] is missing required field 'name'")
        output.append(merged)
    return output


def parse_gpu_pairs(raw: str | None) -> list[str]:
    """Parse a semicolon-separated list such as '0,1;2,3;4,5;6,7'."""
    if raw is None or not str(raw).strip():
        return []
    pairs = [part.strip() for part in str(raw).split(";") if part.strip()]
    if not pairs:
        raise ValueError("EVOCO_GPU_PAIRS is set but contains no GPU pairs")
    for pair in pairs:
        ids = [item.strip() for item in pair.split(",") if item.strip()]
        if len(ids) < 1:
            raise ValueError(f"invalid GPU pair in EVOCO_GPU_PAIRS: {pair!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate GPU id in EVOCO_GPU_PAIRS pair: {pair!r}")
        for gpu_id in ids:
            if not gpu_id.isdigit():
                raise ValueError(f"GPU ids must be integers in EVOCO_GPU_PAIRS: {pair!r}")
    return pairs


def format_run_name(index: int, experiment: Mapping[str, Any], include_gpu: bool = True) -> str:
    parts = [f"{index:02d}_{slug(str(experiment.get('name') or f'run_{index:02d}'))}"]
    dataset_id = experiment.get("dataset_id")
    seed = experiment.get("seed")
    gpu = experiment.get("gpu")
    if dataset_id is not None and slug(str(dataset_id)) != slug(str(experiment.get("name", ""))):
        parts.append(slug(str(dataset_id)))
    if seed is not None:
        parts.append(f"s{seed}")
    if include_gpu and gpu is not None:
        parts.append(f"g{slug(str(gpu))}")
    return "_".join(parts)


def resolve_path(raw: str, *, cwd: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path


def resolve_study_dir(spec: Mapping[str, Any], spec_path: Path) -> Path:
    root = resolve_path(str(spec.get("output_root", "../rag_assets/outputs/experiments")), cwd=Path.cwd())
    study_name = spec.get("study_name")
    if not study_name:
        stamp = datetime.now().strftime("%Y%m%d")
        study_name = f"{stamp}_{spec_path.stem}"
    return root / slug(str(study_name))


def final_round_index(config: Mapping[str, Any]) -> int:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        return 0
    num_rounds = int(training.get("num_rounds", 1) or 1)
    return max(num_rounds - 1, 0)


def train_marker_path(run_dir: Path, config: Mapping[str, Any]) -> Path:
    return run_dir / "metrics" / f"round_{final_round_index(config):03d}.json"


def eval_marker_path(run_dir: Path) -> Path:
    return run_dir / "metrics" / "test_eval.json"


def build_run(
    spec: Mapping[str, Any],
    spec_path: Path,
    experiment: Mapping[str, Any],
    index: int,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    base_config_raw = experiment.get("config", spec.get("base_config", "configs/evoco_popqa_fast.yaml"))
    base_config_path = resolve_path(str(base_config_raw), cwd=Path.cwd())
    base_config = read_yaml(base_config_path)

    overrides = dict(spec.get("overrides", {}) or {})
    overrides.update(experiment.get("overrides", {}) or {})
    if experiment.get("seed") is not None:
        overrides["project.seed"] = int(experiment["seed"])
    if experiment.get("debug_size") is not None:
        overrides["data.debug_size"] = int(experiment["debug_size"])
    if experiment.get("num_rounds") is not None:
        overrides["training.num_rounds"] = int(experiment["num_rounds"])
    if experiment.get("batch_size") is not None:
        overrides["training.batch_size"] = int(experiment["batch_size"])
    if experiment.get("large_batch_size") is not None:
        overrides["training.large_batch_size"] = int(experiment["large_batch_size"])

    config = apply_overrides(base_config, overrides)
    # Fail before launching an expensive run when a dotted override is stale or
    # misspelled. EvoCoConfig performs strict section/key validation.
    EvoCoConfig.from_dict(config)
    study_dir = resolve_study_dir(spec, spec_path)
    run_name = str(experiment.get("run_name") or format_run_name(index, experiment))
    run_dir = study_dir / run_name
    checkpoint_root = resolve_path(
        str(spec.get("checkpoint_root", "../rag_assets/checkpoints/experiments")),
        cwd=Path.cwd(),
    )
    checkpoint_dir = checkpoint_root / slug(str(spec.get("study_name") or spec_path.stem)) / run_name

    if experiment.get("auto_paths", spec.get("auto_paths", True)):
        config.setdefault("project", {})["name"] = str(experiment.get("project_name") or run_name)
        config.setdefault("project", {})["output_dir"] = str(run_dir)
        config.setdefault("models", {})["small_lora_dir"] = str(checkpoint_dir / "small")
        config.setdefault("models", {})["large_lora_dir"] = str(checkpoint_dir / "large")

    config_path = run_dir / "run_config.yaml"
    log_path = run_dir / "train.log"
    eval_log_path = run_dir / "eval.log"
    eval_after_train = bool(experiment.get("eval_after_train", spec.get("eval_after_train", True)))
    train_marker = train_marker_path(run_dir, config)
    completion_marker = eval_marker_path(run_dir) if eval_after_train else train_marker
    status = (
        "exists"
        if run_dir.exists()
        and completion_marker.exists()
        and not overwrite
        else "ready"
    )

    python_executable = str(experiment.get("python", spec.get("python", "python")))
    train_script = str(experiment.get("train_script", spec.get("train_script", DEFAULT_TRAIN_SCRIPT)))
    command = [python_executable, train_script, "--config", str(config_path)]
    for extra in experiment.get("extra_args", spec.get("extra_args", [])) or []:
        command.append(str(extra))
    if experiment.get("resume", spec.get("resume", False)):
        command.append("--resume")

    eval_script = str(experiment.get("eval_script", spec.get("eval_script", DEFAULT_EVAL_SCRIPT)))
    eval_command = [python_executable, eval_script, "--config", str(config_path)]
    for extra in experiment.get("eval_extra_args", spec.get("eval_extra_args", [])) or []:
        eval_command.append(str(extra))

    return {
        "index": index,
        "name": experiment["name"],
        "gpu": experiment.get("gpu"),
        "status": status,
        "eval_after_train": eval_after_train,
        "overwrite": overwrite,
        "base_config": str(base_config_path),
        "run_dir": run_dir,
        "config_path": config_path,
        "log_path": log_path,
        "eval_log_path": eval_log_path,
        "train_marker_path": train_marker,
        "completion_marker_path": completion_marker,
        "checkpoint_dir": checkpoint_dir,
        "config": config,
        "command": command,
        "eval_command": eval_command,
        "overrides": overrides,
    }


def write_manifest(study_dir: Path, runs: list[Mapping[str, Any]], spec_path: Path) -> None:
    manifest = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "spec": str(spec_path.resolve()),
        "runs": [
            {
                "index": run["index"],
                "name": run["name"],
                "gpu": run["gpu"],
                "status": run["status"],
                "eval_after_train": run["eval_after_train"],
                "base_config": run["base_config"],
                "run_dir": str(run["run_dir"]),
                "config_path": str(run["config_path"]),
                "log_path": str(run["log_path"]),
                "eval_log_path": str(run["eval_log_path"]),
                "train_marker_path": str(run["train_marker_path"]),
                "completion_marker_path": str(run["completion_marker_path"]),
                "checkpoint_dir": str(run["checkpoint_dir"]),
                "command": run["command"],
                "eval_command": run["eval_command"] if run["eval_after_train"] else None,
                "overrides": run["overrides"],
            }
            for run in runs
        ],
    }
    write_yaml(study_dir / "launch_manifest.yaml", manifest)


def shell_command(run: Mapping[str, Any]) -> str:
    train_command = quote_command(run["command"])
    eval_command = quote_command(run["eval_command"])
    run_dir = shlex.quote(str(run["run_dir"]))
    log_path = shlex.quote(str(run["log_path"]))
    eval_log_path = shlex.quote(str(run["eval_log_path"]))
    train_marker = shlex.quote(str(run["train_marker_path"]))
    completion_marker = shlex.quote(str(run["completion_marker_path"]))
    env_prefix = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(run['gpu']))} "
        if run.get("gpu") is not None
        else ""
    )
    train_part = f"{env_prefix}{train_command} 2>&1 | tee {log_path}"
    if not run.get("eval_after_train"):
        return f"mkdir -p {run_dir} && {train_part}"
    eval_part = f"{env_prefix}{eval_command} 2>&1 | tee {eval_log_path}"
    post_train_eval = (
        f"if [ -f {completion_marker} ]; then "
        "echo 'final-round test metrics found; skip duplicate eval'; "
        f"else {eval_part}; fi"
    )
    if not run.get("overwrite"):
        eval_only_part = f"echo 'train complete marker found; running eval only' && {eval_part}"
        return f"mkdir -p {run_dir} && if [ -f {train_marker} ]; then {eval_only_part}; else {train_part} && {post_train_eval}; fi"
    return f"mkdir -p {run_dir} && {train_part} && {post_train_eval}"


def write_gpu_scripts(study_dir: Path, runs: list[Mapping[str, Any]]) -> Path:
    by_gpu: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        gpu = "cpu" if run.get("gpu") is None else str(run["gpu"])
        by_gpu.setdefault(gpu, []).append(run)

    tmux_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for gpu, gpu_runs in sorted(by_gpu.items(), key=lambda item: item[0]):
        gpu_slug = slug(gpu)
        script_path = study_dir / f"run_gpu{gpu_slug}.sh"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(Path.cwd()))}",
            "",
        ]
        for run in gpu_runs:
            complete_marker = shlex.quote(str(run["completion_marker_path"]))
            failure_note = f"echo '[{run['index']:02d}] FAILED: {run['run_dir']}'"
            lines.extend(
                [
                    f"echo '[{run['index']:02d}] {run['name']} -> {run['run_dir']}'",
                    f"if [ -f {complete_marker} ]; then echo 'skip complete: {run['run_dir']}'; else",
                    f"  {shell_command(run)} || {failure_note}",
                    "fi",
                    "",
                ]
            )
        script_path.write_text("\n".join(lines), encoding="utf-8")
        script_path.chmod(0o755)
        session = f"evoco_{study_dir.name}_g{gpu_slug}"[:80]
        quoted_session = shlex.quote(session)
        tmux_lines.extend(
            [
                f"if tmux has-session -t {quoted_session} 2>/dev/null; then",
                f"  echo 'tmux session already exists: {session}'",
                "else",
                f"  tmux new -d -s {quoted_session} {shlex.quote('bash ' + str(script_path))}",
                "fi",
                "",
            ]
        )

    tmux_path = study_dir / "launch_tmux.sh"
    tmux_path.write_text("\n".join(tmux_lines) + "\n", encoding="utf-8")
    tmux_path.chmod(0o755)
    return tmux_path


def print_run(run: Mapping[str, Any]) -> None:
    env_prefix = f"CUDA_VISIBLE_DEVICES={run['gpu']} " if run.get("gpu") is not None else ""
    print(f"[{run['index']:02d}] {run['name']} -> {run['run_dir']}")
    print(f"     status: {run['status']}, gpu: {run.get('gpu')}")
    print(f"     config: {run['config_path']}")
    print(f"     checkpoint_dir: {run['checkpoint_dir']}")
    print(f"     command: {env_prefix}{quote_command(run['command'])}")
    if run.get("eval_after_train"):
        print(f"     eval: {env_prefix}{quote_command(run['eval_command'])}")
    print(f"     shell: {shell_command(run)}")


def stream_command(command: list[str], log_path: Path, env: Mapping[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def launch_run(run: Mapping[str, Any]) -> int:
    if run["status"] == "exists":
        print(f"SKIP existing run: {run['run_dir']}")
        return 0

    run_dir: Path = run["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run["config_path"], run["config"])
    env = os.environ.copy()
    if run.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(run["gpu"])
    train_done = Path(run["train_marker_path"]).exists()
    if train_done and run.get("eval_after_train") and not run.get("overwrite"):
        print(f"SKIP training; found train marker: {run['train_marker_path']}")
        train_code = 0
    else:
        train_code = stream_command(run["command"], run["log_path"], env)
    if train_code != 0 or not run.get("eval_after_train"):
        return train_code
    if Path(run["completion_marker_path"]).exists():
        print("Final-round test metrics found; skip duplicate evaluation.")
        return 0
    return stream_command(run["eval_command"], run["eval_log_path"], env)


def launch_tmux(tmux_path: Path) -> None:
    subprocess.run(["bash", str(tmux_path)], check=True)
    print(f"Started tmux queues with: bash {tmux_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML experiment spec.")
    parser.add_argument("--launch", action="store_true", help="Actually run experiments sequentially.")
    parser.add_argument(
        "--launch-tmux",
        "--tmux",
        action="store_true",
        help="Generate scripts and start all per-GPU queues with bash launch_tmux.sh.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Do not skip existing run directories.")
    parser.add_argument("--no-gpu-scripts", action="store_true", help="Do not write run_gpu*.sh scripts.")
    args = parser.parse_args()
    if args.launch and args.launch_tmux:
        parser.error("--launch and --launch-tmux are mutually exclusive")
    if args.launch_tmux and args.no_gpu_scripts:
        parser.error("--launch-tmux requires GPU scripts; remove --no-gpu-scripts")

    spec_path = Path(args.spec).resolve()
    spec = read_yaml(spec_path)
    experiments = experiment_list(spec)
    runs = [
        build_run(spec, spec_path, experiment, index, overwrite=args.overwrite)
        for index, experiment in enumerate(experiments)
    ]
    study_dir = resolve_study_dir(spec, spec_path)
    study_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["run_dir"].mkdir(parents=True, exist_ok=True)
        write_yaml(run["config_path"], run["config"])
        print_run(run)
    write_manifest(study_dir, runs, spec_path)
    tmux_path = study_dir / "launch_tmux.sh"
    if not args.no_gpu_scripts:
        tmux_path = write_gpu_scripts(study_dir, runs)

    if args.launch_tmux:
        launch_tmux(tmux_path)
        return

    if not args.launch:
        print(f"Dry run complete. Study directory: {study_dir}")
        print("Use --launch to execute these runs sequentially.")
        if not args.no_gpu_scripts:
            print(f"Per-GPU scripts: {study_dir}/run_gpu*.sh")
            print(f"Tmux launcher: {study_dir}/launch_tmux.sh")
            print(f"Use --launch-tmux to start tmux queues directly.")
        return

    failures = []
    for run in runs:
        code = launch_run(run)
        if code != 0:
            failures.append((run["name"], code))
            print(f"FAILED {run['name']} with exit code {code}")
    succeeded = len(runs) - len(failures)
    print(f"\nLaunch complete: {succeeded}/{len(runs)} runs succeeded.")
    if failures:
        for name, code in failures:
            print(f"  FAILED {name} (exit {code})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
