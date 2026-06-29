"""经验回放池（开发文档 §5.9、§12）。

每个样本每轮写一条 JSONL；支持按 round / failure_type / audit_trust_weight 过滤，
采样 high-confidence positives 与 hard negatives，防止低质量自训练数据无限累积。
只用标准库，不依赖 jsonlines。
"""

from __future__ import annotations

import json
import glob
import os
import random
from collections import Counter
from typing import Iterable, Optional

from .schemas import ReplayExperience


class ReplayBuffer:
    def __init__(self, root: str = "outputs/replay"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # ---- 路径 ----
    def round_path(self, round_id: int) -> str:
        return os.path.join(self.root, f"round_{round_id:03d}.jsonl")

    @property
    def all_path(self) -> str:
        return os.path.join(self.root, "all.jsonl")

    # ---- 写 ----
    def write(self, experiences: Iterable[ReplayExperience], round_id: int) -> int:
        rp = self.round_path(round_id)
        n = 0
        with open(rp, "w", encoding="utf-8") as fr:
            for exp in experiences:
                line = json.dumps(exp.to_dict(), ensure_ascii=False)
                fr.write(line + "\n")
                n += 1
        self.rebuild_all()
        return n

    def rebuild_all(self) -> None:
        """Rebuild all.jsonl from round_*.jsonl to avoid duplicate appends."""
        round_paths = sorted(glob.glob(os.path.join(self.root, "round_*.jsonl")))
        with open(self.all_path, "w", encoding="utf-8") as fa:
            for path in round_paths:
                with open(path, "r", encoding="utf-8") as fr:
                    for line in fr:
                        if line.strip():
                            fa.write(line)

    def reset(self) -> None:
        for path in glob.glob(os.path.join(self.root, "round_*.jsonl")) + [self.all_path]:
            if os.path.exists(path):
                os.remove(path)

    # ---- 读 ----
    def read(self, round_id: Optional[int] = None) -> list[ReplayExperience]:
        path = self.all_path if round_id is None else self.round_path(round_id)
        out: list[ReplayExperience] = []
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(ReplayExperience.from_dict(json.loads(line)))
        return out

    # ---- 过滤 ----
    @staticmethod
    def filter_by_failure_type(exps: list[ReplayExperience], failure_type: str) -> list[ReplayExperience]:
        return [
            e for e in exps
            if (e.training_targets.get("failure_type")
                or e.audit.get("failure_type")) == failure_type
        ]

    @staticmethod
    def filter_by_trust(exps: list[ReplayExperience], min_weight: float) -> list[ReplayExperience]:
        return [
            e for e in exps
            if e.verification.get("audit_trust_weight", 0.0) >= min_weight
        ]

    # ---- 采样 ----
    @staticmethod
    def sample_small_training_pairs(
        exps: list[ReplayExperience],
        min_trust: float = 0.5,
    ) -> list[dict]:
        """产出小模型训练样本：每条含 positive / negative doc_ids。

        新版 rule-verifier 标签由 gold/evidence 规则直接产生，不依赖大模型审计
        可信度；旧 replay 仍保留 min_trust 过滤以避免兼容路径引入噪声。
        """
        pairs = []
        for e in exps:
            source = e.training_targets.get("small_target_source")
            if (
                source != "gold_rule_verifier"
                and e.verification.get("audit_trust_weight", 0.0) < min_trust
            ):
                continue
            tt = e.training_targets
            pos = tt.get("small_positive_doc_ids", [])
            neg = tt.get("small_negative_doc_ids", [])
            if not pos and not neg:
                continue
            pairs.append({
                "sample_id": e.sample_id,
                "question": e.question,
                "documents": e.documents,
                "positive_doc_ids": pos,
                "negative_doc_ids": neg,
                "positive_doc_weights": tt.get("small_positive_doc_weights", {}),
                "negative_doc_weights": tt.get("small_negative_doc_weights", {}),
                "hard_negative_doc_ids": tt.get("small_hard_negative_doc_ids", []),
                "action_target": tt.get("small_action_target"),
            })
        return pairs

    @staticmethod
    def sample_large_sft(exps: list[ReplayExperience]) -> list[ReplayExperience]:
        return [e for e in exps if e.training_targets.get("large_sft_eligible")]

    @staticmethod
    def downsample_noisy(
        exps: list[ReplayExperience],
        max_low_trust_ratio: float = 0.3,
        min_trust: float = 0.5,
        seed: int = 42,
    ) -> list[ReplayExperience]:
        """限制低质量伪标签比例，避免自训练噪声累积（开发文档 §13.2、风险四）。"""
        rng = random.Random(seed)
        high = [e for e in exps if e.verification.get("audit_trust_weight", 0.0) >= min_trust]
        low = [e for e in exps if e.verification.get("audit_trust_weight", 0.0) < min_trust]
        if not high:
            return exps
        max_low = int(len(high) * max_low_trust_ratio)
        if len(low) > max_low:
            low = rng.sample(low, max_low)
        merged = high + low
        rng.shuffle(merged)
        return merged

    # ---- 统计 ----
    @staticmethod
    def attribution_distribution(exps: list[ReplayExperience]) -> dict:
        """Return counts by responsibility-attribution case."""
        counter = Counter()
        for e in exps:
            case = e.training_targets.get("attribution_case") or e.rewards.get("attribution_case") or "unknown"
            counter[case] += 1
        return dict(counter)

    @staticmethod
    def credit_assignment_summary(exps: list[ReplayExperience]) -> dict:
        """Metrics for ECR-2: answer-only retriever mis-credit and quadrant counts."""
        total = len(exps)
        wrong = sum(
            1 for e in exps
            if e.training_targets.get("wrong_retriever_reward_if_answer_only")
        )
        filtered = sum(
            1 for e in exps
            if e.training_targets.get("do_not_reward_retriever_reason")
        )
        return {
            "num_experiences": total,
            "attribution_case_distribution": ReplayBuffer.attribution_distribution(exps),
            "wrong_retriever_reward_count": wrong,
            "wrong_retriever_reward_rate": (wrong / total) if total else 0.0,
            "retriever_positive_filtered_count": filtered,
            "retriever_positive_filtered_rate": (filtered / total) if total else 0.0,
        }

    @staticmethod
    def trust_summary(exps: list[ReplayExperience], low_trust_threshold: float = 0.5) -> dict:
        """Metrics for ECR-3: audit reliability and trust-weight distribution."""
        total = len(exps)
        if not total:
            return {
                "num_experiences": 0,
                "audit_json_valid_rate": 0.0,
                "audit_trust_weight_mean": 0.0,
                "low_trust_rate": 0.0,
                "audit_self_consistency_mean": None,
            }

        json_valid = [1.0 if e.verification.get("json_valid") else 0.0 for e in exps]
        trusts = [float(e.verification.get("audit_trust_weight", 0.0)) for e in exps]
        low = [1.0 if t < low_trust_threshold else 0.0 for t in trusts]
        consistency = []
        for e in exps:
            meta = e.audit.get("audit_metadata") or {}
            value = meta.get("self_consistency")
            if value is not None:
                try:
                    consistency.append(float(value))
                except (TypeError, ValueError):
                    pass
        return {
            "num_experiences": total,
            "audit_json_valid_rate": sum(json_valid) / total,
            "audit_trust_weight_mean": sum(trusts) / total,
            "low_trust_rate": sum(low) / total,
            "audit_self_consistency_mean": (
                sum(consistency) / len(consistency) if consistency else None
            ),
        }
