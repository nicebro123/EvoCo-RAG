"""消融实验跑批（开发文档 §7.2 实验矩阵）。

对同一份数据，按不同 ablation 配置依次跑协同进化，汇总各实验最终指标，便于
对比"审计 / 动态 action / reward 拆解 / 单边训练"各自的贡献。

每个实验单独加载模型和 LoRA，避免不同消融实验互相污染 adapter 状态。

用法（H20）:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_ablations.py --config configs/evoco_popqa.yaml
快速逻辑校验（无 GPU，纯启发式、不训练、不审计）:
    python scripts/run_ablations.py --config configs/debug.yaml --no_models
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.config import EvoCoConfig
from evoco_rag.data import load_train_samples
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer
from evoco_rag.weights import prepare_weight_layout

# 实验矩阵：每项是对 ablation 字段的覆盖（未列字段保持 True）
EXPERIMENTS = {
    "baseline_current_corag": dict(use_evidence_audit=False, use_action_policy=False,
                                   use_decomposed_reward=False),
    "evoco_no_audit":        dict(use_evidence_audit=False),
    "evoco_no_action":       dict(use_action_policy=False),
    "evoco_answer_only_reward": dict(use_decomposed_reward=False),
    "evoco_small_only":      dict(train_large_lora=False),
    "evoco_large_only":      dict(train_small_lora=False),
    "evoco_full":            dict(),
}


def _apply_ablation(cfg: EvoCoConfig, overrides: dict, name: str, no_models: bool):
    c = copy.deepcopy(cfg)
    for k, v in overrides.items():
        setattr(c.ablation, k, v)
    if no_models:
        # 无模型：不能审计、不能训练，只验证逻辑链路与开关接线
        c.ablation.use_evidence_audit = False
        c.ablation.train_small_lora = False
        c.ablation.train_large_lora = False
    c.output_dir = os.path.join(cfg.output_dir, "ablations", name)
    c.models.small_lora_dir = os.path.join(c.output_dir, "checkpoints", "small")
    c.models.large_lora_dir = os.path.join(c.output_dir, "checkpoints", "large")
    os.makedirs(c.output_dir, exist_ok=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evoco_popqa.yaml")
    ap.add_argument("--no_models", action="store_true",
                    help="不加载模型，纯启发式逻辑校验")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定实验名")
    args = ap.parse_args()

    base = EvoCoConfig.load(args.config)
    prepare_weight_layout(base, create=True)
    samples = load_train_samples(base.data.train_path, base.data.dataset_name, base.data.debug_size)
    print(f"loaded {len(samples)} samples")

    summary = {}
    names = args.only or list(EXPERIMENTS.keys())
    for name in names:
        overrides = EXPERIMENTS[name]
        cfg = _apply_ablation(base, overrides, name, args.no_models)
        print(f"\n===== ablation: {name}  ({overrides or 'full'}) =====")

        small_policy = large_auditor = small_trainer = large_trainer = None
        if not args.no_models:
            from evoco_rag.large_model import LargeGeneratorAuditor
            from evoco_rag.small_model import SmallRagPolicy
            from evoco_rag.trainers.large_trainer import LargeTrainer
            from evoco_rag.trainers.small_trainer import SmallTrainer
            small_policy = SmallRagPolicy(cfg.models.small_base_path, use_lora=True)
            large_auditor = LargeGeneratorAuditor(cfg.models.large_base_path, use_lora=True,
                                                  use_4bit=cfg.models.use_4bit)
            small_trainer = SmallTrainer(small_policy, lr=cfg.training.small_lr,
                                         batch_size=cfg.training.batch_size)
            large_trainer = LargeTrainer(large_auditor, lr=cfg.training.large_lr)

        evaluator = Evaluator(cfg, small_policy, large_auditor)
        trainer = CoevolutionTrainer(
            cfg, small_policy, large_auditor,
            small_trainer if cfg.ablation.train_small_lora else None,
            large_trainer if cfg.ablation.train_large_lora else None,
            evaluator)
        stats = trainer.run(samples)
        last_eval = stats[-1].get("eval", {}) if stats else {}
        summary[name] = last_eval
        print(f"{name} final eval: {json.dumps(last_eval, ensure_ascii=False)}")

    out = os.path.join(base.output_dir, "ablations", "summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsummary -> {out}")
    # 打印对比表
    keys = ["accuracy", "evidence_support_rate", "unsupported_answer_rate",
            "citation_correctness", "avg_selected_docs"]
    print("\n{:<26}".format("experiment") + "".join(f"{k:>22}" for k in keys))
    for name, m in summary.items():
        row = "".join(f"{(m.get(k) if m.get(k) is not None else '-'):>22}" for k in keys)
        print("{:<26}".format(name) + row)


if __name__ == "__main__":
    main()
