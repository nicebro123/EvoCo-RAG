"""测试集评估入口（开发文档 §7、§9.4）。

gold answers 不进入大模型 prompt，只用于离线指标。需要 GPU + 模型权重。
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_evoco.py --config configs/evoco_popqa.yaml \
        --small_lora ../rag_assets/checkpoints/evoco_popqa/small/round_002 \
        --large_lora ../rag_assets/checkpoints/evoco_popqa/large/round_002
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.config import EvoCoConfig
from evoco_rag.data import load_test_samples
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.large_model import LargeGeneratorAuditor
from evoco_rag.small_model import SmallRagPolicy
from evoco_rag.weights import resolve_adapter_for_loading, write_weight_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evoco_popqa.yaml")
    ap.add_argument("--small_lora", default=None)
    ap.add_argument("--large_lora", default=None)
    args = ap.parse_args()

    cfg = EvoCoConfig.load(args.config)
    write_weight_manifest(cfg, cfg.output_dir)
    samples = load_test_samples(cfg.data.test_path, cfg.data.dataset_name, cfg.data.debug_size)
    print(f"loaded {len(samples)} test samples")
    small_lora = resolve_adapter_for_loading(args.small_lora or cfg.models.small_lora_dir)
    large_lora = resolve_adapter_for_loading(args.large_lora or cfg.models.large_lora_dir)
    print(f"eval small LoRA: {small_lora or 'none'}")
    print(f"eval large LoRA: {large_lora or 'none'}")

    small_policy = SmallRagPolicy(base_path=cfg.models.small_base_path,
                                  lora_dir=small_lora, use_lora=False)
    large_auditor = LargeGeneratorAuditor(base_path=cfg.models.large_base_path,
                                          lora_dir=large_lora, use_lora=False,
                                          use_4bit=cfg.models.use_4bit)
    evaluator = Evaluator(cfg, small_policy, large_auditor)
    metrics = evaluator.run_inference(samples)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    out = os.path.join(cfg.output_dir, "metrics", "test_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
