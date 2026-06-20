"""完整协同进化训练入口（开发文档 §6 阶段6、§13.3）。

只负责 CLI 与 trainer 调用，训练逻辑都在 CoevolutionTrainer。需要 GPU + 模型权重，
在 H20 机器上运行：
    CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/evoco_popqa.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.config import EvoCoConfig
from evoco_rag.data import load_test_samples, load_train_samples
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.large_model import LargeGeneratorAuditor
from evoco_rag.small_model import SmallRagPolicy
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer
from evoco_rag.trainers.large_trainer import LargeTrainer
from evoco_rag.trainers.small_trainer import SmallTrainer
from evoco_rag.weights import (
    adapter_for_round,
    completed_training_rounds,
    latest_checkpoint_round,
    prepare_weight_layout,
    write_weight_manifest,
)


def publish_final_round_metrics(output_dir: str, stats: list[dict]) -> int:
    """Expose the already-computed final-round test result under canonical names."""
    if not stats:
        raise RuntimeError("training completed without any round statistics")
    import shutil

    final_round = int(stats[-1]["round"])
    metrics_dir = os.path.join(output_dir, "metrics")
    shutil.copyfile(
        os.path.join(metrics_dir, f"test_eval_round_{final_round:03d}.json"),
        os.path.join(metrics_dir, "test_eval.json"),
    )
    shutil.copyfile(
        os.path.join(metrics_dir, f"test_predictions_round_{final_round:03d}.jsonl"),
        os.path.join(metrics_dir, "test_predictions.jsonl"),
    )
    return final_round


def resolve_training_state(cfg, resume: bool) -> tuple[int, str | None, str | None]:
    """Resolve only checkpoints backed by completed per-round test artifacts."""
    latest_small = latest_checkpoint_round(cfg.models.small_lora_dir)
    latest_large = latest_checkpoint_round(cfg.models.large_lora_dir)
    available_checkpoints = [r for r in (latest_small, latest_large) if r is not None]
    completed = completed_training_rounds(cfg.output_dir)

    if not resume:
        if available_checkpoints or completed:
            raise SystemExit(
                "output/checkpoint roots already contain completed or partial rounds. "
                "Use --resume to continue, or change project.output_dir / "
                "models.*_lora_dir for a fresh run.")
        return 0, None, None

    if not completed:
        if not available_checkpoints:
            raise SystemExit(
                "--resume was set, but no checkpoint or completed round was found.")
        # Checkpoints without an authoritative round metric are from an
        # interrupted round. Ignore them and safely replay round 0 from base.
        return 0, None, None

    last_completed = completed[-1]
    if completed != list(range(last_completed + 1)):
        raise SystemExit(
            f"completed round metrics are not contiguous: {completed}")
    start_round = last_completed + 1
    if start_round >= cfg.training.num_rounds:
        raise SystemExit(
            f"resume state is already complete at round {last_completed}; "
            f"training.num_rounds={cfg.training.num_rounds} leaves no remaining rounds.")

    small_init = None
    if cfg.ablation.train_small_lora:
        small_init = adapter_for_round(cfg.models.small_lora_dir, last_completed)
        if small_init is None:
            raise SystemExit(
                f"round {last_completed} is marked complete but its small adapter is missing.")
    large_init = None
    if cfg.ablation.train_large_lora:
        large_init = adapter_for_round(cfg.models.large_lora_dir, last_completed)
        if large_init is None:
            raise SystemExit(
                f"round {last_completed} is marked complete but its large adapter is missing.")
    return start_round, small_init, large_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evoco_popqa.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="从 checkpoint root 下最新 round_* LoRA 继续；不传则初始化新 LoRA")
    args = ap.parse_args()

    cfg = EvoCoConfig.load(args.config)
    layout = prepare_weight_layout(cfg, create=True)
    write_weight_manifest(cfg, cfg.output_dir)
    # 落盘所用配置，保证可复现（开发文档 §12）
    import shutil
    shutil.copy(args.config, os.path.join(cfg.output_dir, "used_config.yaml"))

    samples = load_train_samples(cfg.data.train_path, cfg.data.dataset_name, cfg.data.debug_size)
    print(f"loaded {len(samples)} train samples")
    # 每轮训练后做真实泛化评估的测试子集（gold 不进 prompt）
    eval_cap = cfg.data.eval_size if cfg.data.eval_size is not None else cfg.data.debug_size
    test_samples = load_test_samples(cfg.data.test_path, cfg.data.dataset_name, eval_cap)
    print(f"loaded {len(test_samples)} test samples for per-round generalization eval")
    if not test_samples:
        raise SystemExit(
            "per-round test evaluation is required, but the configured test split is empty.")
    start_round, small_init, large_init = resolve_training_state(cfg, args.resume)
    print(f"base small model: {layout['small_base_path']}")
    print(f"base large model: {layout['large_base_path']}")
    print(f"init small LoRA: {small_init or 'fresh adapter'}")
    print(f"init large LoRA: {large_init or 'fresh adapter'}")
    print(f"start round: {start_round}")

    small_policy = SmallRagPolicy(
        base_path=cfg.models.small_base_path,
        lora_dir=small_init,
        use_lora=True,
        use_policy_heads=cfg.small_policy.use_policy_heads)
    large_auditor = LargeGeneratorAuditor(
        base_path=cfg.models.large_base_path,
        lora_dir=large_init,
        use_lora=True, use_4bit=cfg.models.use_4bit,
        max_prompt_length=cfg.runtime.max_prompt_length,
        max_completion_length=cfg.runtime.max_completion_length,
        candidate_doc_char_limit=cfg.runtime.candidate_doc_char_limit,
        num_audit_candidates=cfg.runtime.num_audit_candidates,
        audit_batch_size=cfg.runtime.audit_batch_size,
        audit_temperature=cfg.runtime.audit_temperature)

    small_trainer = SmallTrainer(
        small_policy,
        lr=cfg.training.small_lr,
        batch_size=cfg.training.batch_size,
        evidence_loss_weight=cfg.small_policy.evidence_loss_weight,
        action_loss_weight=cfg.small_policy.action_loss_weight,
        calibration_loss_weight=cfg.small_policy.calibration_loss_weight)
    large_trainer = LargeTrainer(
        large_auditor,
        lr=cfg.training.large_lr,
        max_prompt_length=cfg.runtime.max_prompt_length,
        max_completion_length=cfg.runtime.max_completion_length,
        batch_size=cfg.training.large_batch_size)
    evaluator = Evaluator(cfg, small_policy, large_auditor, test_samples=test_samples)

    trainer = CoevolutionTrainer(cfg, small_policy, large_auditor,
                                 small_trainer, large_trainer, evaluator)
    stats = trainer.run(samples, start_round=start_round)
    incomplete_rounds = [
        stat.get("round") for stat in stats
        if stat.get("eval_source") != "test_generalization"
        or not stat.get("per_round_test_completed")
    ]
    if incomplete_rounds:
        raise RuntimeError(
            f"per-round test evaluation did not complete for rounds: {incomplete_rounds}")
    final_round = publish_final_round_metrics(cfg.output_dir, stats)
    print(f"published round {final_round} test evaluation as final metrics")
    print("co-evolution finished:")
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
