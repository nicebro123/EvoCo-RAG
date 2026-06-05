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
from evoco_rag.data import load_train_samples
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.large_model import LargeGeneratorAuditor
from evoco_rag.small_model import SmallRagPolicy
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer
from evoco_rag.trainers.large_trainer import LargeTrainer
from evoco_rag.trainers.small_trainer import SmallTrainer
from evoco_rag.weights import (
    latest_checkpoint_round,
    prepare_weight_layout,
    resolve_adapter_for_loading,
    write_weight_manifest,
)


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
    small_init = resolve_adapter_for_loading(cfg.models.small_lora_dir) if args.resume else None
    large_init = resolve_adapter_for_loading(cfg.models.large_lora_dir) if args.resume else None
    latest_small_round = latest_checkpoint_round(cfg.models.small_lora_dir)
    latest_large_round = latest_checkpoint_round(cfg.models.large_lora_dir)
    if not args.resume and (latest_small_round is not None or latest_large_round is not None):
        raise SystemExit(
            "checkpoint roots already contain round_* adapters. "
            "Use --resume to continue, or change project.output_dir / models.*_lora_dir for a fresh run."
        )
    start_round = 0
    if args.resume:
        latest_rounds = [r for r in (latest_small_round, latest_large_round) if r is not None]
        if not latest_rounds:
            raise SystemExit("--resume was set, but no complete round_* LoRA adapter was found.")
        start_round = max(latest_rounds) + 1
        if start_round >= cfg.training.num_rounds:
            raise SystemExit(
                f"resume checkpoint is already at round {start_round - 1}; "
                f"training.num_rounds={cfg.training.num_rounds} leaves no remaining rounds."
            )
    print(f"base small model: {layout['small_base_path']}")
    print(f"base large model: {layout['large_base_path']}")
    print(f"init small LoRA: {small_init or 'fresh adapter'}")
    print(f"init large LoRA: {large_init or 'fresh adapter'}")
    print(f"start round: {start_round}")

    small_policy = SmallRagPolicy(
        base_path=cfg.models.small_base_path,
        lora_dir=small_init,
        use_lora=True)
    large_auditor = LargeGeneratorAuditor(
        base_path=cfg.models.large_base_path,
        lora_dir=large_init,
        use_lora=True, use_4bit=cfg.models.use_4bit,
        max_prompt_length=cfg.runtime.max_prompt_length,
        max_completion_length=cfg.runtime.max_completion_length,
        candidate_doc_char_limit=cfg.runtime.candidate_doc_char_limit,
        num_audit_candidates=cfg.runtime.num_audit_candidates,
        audit_temperature=cfg.runtime.audit_temperature)

    small_trainer = SmallTrainer(small_policy, lr=cfg.training.small_lr,
                                 batch_size=cfg.training.batch_size)
    large_trainer = LargeTrainer(
        large_auditor,
        lr=cfg.training.large_lr,
        max_prompt_length=cfg.runtime.max_prompt_length,
        max_completion_length=cfg.runtime.max_completion_length)
    evaluator = Evaluator(cfg, small_policy, large_auditor)

    trainer = CoevolutionTrainer(cfg, small_policy, large_auditor,
                                 small_trainer, large_trainer, evaluator)
    stats = trainer.run(samples, start_round=start_round)
    print("co-evolution finished:")
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
