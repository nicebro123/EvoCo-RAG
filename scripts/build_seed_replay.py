"""阶段1：生成 seed replay（开发文档 §6 阶段1）。

不加载任何模型，只用启发式（seed_labels / 文档原序）产出证据合约，并跑通
verify → decomposed reward → replay 写盘的全链路。可在纯 CPU 环境运行，用于
验证 schema、合约与 replay 的正确性。

用法:
    python scripts/build_seed_replay.py --config configs/debug.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.config import EvoCoConfig
from evoco_rag.contract import build_contract
from evoco_rag.data import load_train_samples
from evoco_rag.rewards import build_training_targets, compute_decomposed_reward
from evoco_rag.replay_buffer import ReplayBuffer
from evoco_rag.schemas import LargeAudit, ReplayExperience
from evoco_rag.verifier import verify
from evoco_rag.weights import prepare_weight_layout, write_weight_manifest


def seed_score(sample, doc_id):
    """用 seed_labels 的历史正例比例当作粗打分；没有则用逆序。"""
    if sample.seed_labels and doc_id < len(sample.seed_labels):
        label = sample.seed_labels[doc_id]
        if isinstance(label, list) and label:
            ones = sum(1 for x in label if str(x) == "1")
            return ones / len(label)
    return -doc_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/debug.yaml")
    args = ap.parse_args()

    cfg = EvoCoConfig.load(args.config)
    prepare_weight_layout(cfg, create=True)
    write_weight_manifest(cfg, cfg.output_dir)
    samples = load_train_samples(cfg.data.train_path, cfg.data.dataset_name, cfg.data.debug_size)
    print(f"loaded {len(samples)} train samples")

    rb = ReplayBuffer(root=os.path.join(cfg.output_dir, "replay"))
    experiences = []
    for s in samples:
        ranked = sorted(
            [{"doc_id": d["doc_id"], "score": seed_score(s, d["doc_id"])} for d in s.documents],
            key=lambda x: x["score"], reverse=True)
        contract = build_contract(s, ranked, round_id=0, top_k=cfg.contract.top_k,
                                  max_selected_docs=cfg.contract.max_selected_docs)
        # 阶段1 大模型只给占位审计（用 top1 当引用、答案留空）
        audit = LargeAudit(sample_id=s.sample_id, round=0, final_answer="",
                           used_doc_ids=contract.selected_doc_ids()[:1])
        v = verify(s, contract, audit, json_valid=True)
        r = compute_decomposed_reward(s, contract, audit, v, cfg.reward)
        t = build_training_targets(s, contract, audit, v, r)
        experiences.append(ReplayExperience(
            sample_id=s.sample_id, round=0, question=s.question, answers=s.answers,
            documents=s.documents, contract=contract.to_dict(), audit=audit.to_dict(),
            verification=v.to_dict(), rewards=r.to_dict(), training_targets=t))

    n = rb.write(experiences, round_id=0)
    for name, getter in (("contracts", lambda e: e.contract), ("audits", lambda e: e.audit)):
        out = os.path.join(cfg.output_dir, name, "round_000.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for exp in experiences:
                import json
                f.write(json.dumps(getter(exp), ensure_ascii=False) + "\n")
    print(f"wrote {n} experiences -> {rb.round_path(0)}")


if __name__ == "__main__":
    main()
