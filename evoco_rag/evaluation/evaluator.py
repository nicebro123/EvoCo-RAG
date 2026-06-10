"""评估器（开发文档 §7、§9.4）。

两种用法：
  - evaluate(round_id): 离线读取该轮 replay，计算 §7.1 全部指标（无需模型）。
  - run_inference(test_samples): 用当前小/大模型在测试集上跑一遍（gold 不进 prompt），
    生成审计经验再算指标。需要模型，torch 在调用时才用到。
"""

from __future__ import annotations

import json
import os

from .metrics import compute_metrics
from ..replay_buffer import ReplayBuffer


class Evaluator:
    def __init__(self, config, small_policy=None, large_auditor=None, test_samples=None):
        self.cfg = config
        self.small = small_policy
        self.large = large_auditor
        # 真实泛化评估用的测试样本（gold 不进 prompt）。None 表示不做泛化评估。
        self.test_samples = list(test_samples) if test_samples is not None else None
        self.replay = ReplayBuffer(root=os.path.join(config.output_dir, "replay"))

    def can_generalize(self) -> bool:
        """是否具备做真实泛化评估的条件：有测试样本 + 小/大模型。"""
        return bool(self.test_samples) and self.small is not None and self.large is not None

    def evaluate(self, round_id: int) -> dict:
        """训练集诊断：读取本轮 replay（show_gold=True 教师审计）计算指标。"""
        exps = self.replay.read(round_id)
        metrics = compute_metrics(exps)
        out_dir = os.path.join(self.cfg.output_dir, "metrics")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"train_eval_round_{round_id:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return metrics

    def evaluate_generalization(self, round_id: int) -> dict | None:
        """真实泛化评估：在测试集上用当前模型推理（show_gold=False）。

        无测试样本或模型时返回 None（由调用方决定是否回退到训练集诊断）。
        """
        if not self.can_generalize():
            return None
        metrics = self.run_inference(self.test_samples, round_id=round_id)
        out_dir = os.path.join(self.cfg.output_dir, "metrics")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"test_eval_round_{round_id:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return metrics

    def run_inference(self, test_samples, round_id: int = 0) -> dict:
        """测试集推理：gold answers 不可见（show_gold=False）。"""
        from ..verifier import verify
        from ..rewards import build_training_targets, compute_decomposed_reward
        from ..schemas import ReplayExperience, RetrievalAction

        samples = list(test_samples)
        contracts = []
        for sample in samples:
            contract = self.small.build_contract(
                sample, round_id=round_id, top_k=self.cfg.contract.top_k,
                high_conf_threshold=self.cfg.contract.high_conf_threshold,
                answer_now_margin=self.cfg.contract.answer_now_margin,
                max_selected_docs=self.cfg.contract.max_selected_docs,
                action_mode=self.cfg.contract.action_mode,
                policy_action_min_conf=self.cfg.contract.policy_action_min_conf)
            if not self.cfg.ablation.use_action_policy:
                contract.retrieval_action = RetrievalAction.ANSWER_NOW
            contracts.append(contract)

        batch_size = max(1, int(getattr(self.cfg.runtime, "audit_batch_size", 1)))
        if hasattr(self.large, "generate_audit_batch"):
            audits = self.large.generate_audit_batch(
                samples,
                contracts,
                show_gold=False,
                round_id=round_id,
                batch_size=batch_size,
            )
            if len(audits) != len(samples):
                raise RuntimeError("large auditor returned fewer batch audits than samples")
        else:
            audits = [
                self.large.generate_audit(
                    sample, contract, show_gold=False, round_id=round_id)
                for sample, contract in zip(samples, contracts)
            ]

        records = []
        for sample, contract, (audit, json_valid) in zip(samples, contracts, audits):
            v = verify(sample, contract, audit, json_valid=json_valid)
            r = compute_decomposed_reward(sample, contract, audit, v, self.cfg.reward)
            t = build_training_targets(sample, contract, audit, v, r)
            records.append(ReplayExperience(
                sample_id=sample.sample_id, round=round_id,
                question=sample.question, answers=sample.answers,
                documents=sample.documents, contract=contract.to_dict(),
                audit=audit.to_dict(), verification=v.to_dict(),
                rewards=r.to_dict(), training_targets=t))
        return compute_metrics(records)
