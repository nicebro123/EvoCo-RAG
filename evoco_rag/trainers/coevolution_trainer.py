"""协同进化主循环（开发文档 §5.12）。

调度一整轮：小模型出合约 → 大模型审计 → 规则验证 → 分解 reward → 写 replay
→ 训练小模型 LoRA → 训练大模型 LoRA → 评估 → 存 checkpoint。
支持消融开关（ablation）与 resume。
"""

from __future__ import annotations

import json
import os

from ..auditor import build_audit_prompt  # noqa: F401  (供大模型 SFT 复用)
from ..contract import build_contract
from ..rewards import build_training_targets, compute_decomposed_reward
from ..replay_buffer import ReplayBuffer
from ..schemas import ReplayExperience
from ..verifier import verify
from ..weights import checkpoint_round_dir, prepare_weight_layout, write_weight_manifest


class CoevolutionTrainer:
    def __init__(self, config, small_policy, large_auditor,
                 small_trainer=None, large_trainer=None, evaluator=None):
        self.cfg = config
        self.small = small_policy
        self.large = large_auditor
        self.small_trainer = small_trainer
        self.large_trainer = large_trainer
        self.evaluator = evaluator
        self.replay = ReplayBuffer(root=os.path.join(config.output_dir, "replay"))
        prepare_weight_layout(config, create=True)
        write_weight_manifest(config, config.output_dir)

    @staticmethod
    def _write_jsonl(path: str, records: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------- 单样本经验生成
    def make_experience(self, sample, round_id: int) -> ReplayExperience:
        cfg = self.cfg
        # 1) 小模型证据合约
        if self.small is not None:
            contract = self.small.build_contract(
                sample, round_id=round_id, top_k=cfg.contract.top_k,
                high_conf_threshold=cfg.contract.high_conf_threshold,
                answer_now_margin=cfg.contract.answer_now_margin,
                max_selected_docs=cfg.contract.max_selected_docs)
        else:
            # 无模型的纯逻辑回放（测试/调试用）：按文档原序当作打分
            ranked = [{"doc_id": d["doc_id"], "score": -i}
                      for i, d in enumerate(sample.documents)]
            contract = build_contract(sample, ranked, round_id=round_id,
                                      top_k=cfg.contract.top_k)

        # 消融：关闭动态 action policy → 固定 answer_now（等价固定 top-k）
        if not cfg.ablation.use_action_policy:
            from ..schemas import RetrievalAction
            contract.retrieval_action = RetrievalAction.ANSWER_NOW

        # 2) 大模型审计（消融：可关闭，用占位审计）
        json_valid = True
        if cfg.ablation.use_evidence_audit and self.large is not None:
            audit, json_valid = self.large.generate_audit(
                sample, contract, show_gold=True, round_id=round_id)
        else:
            from ..schemas import LargeAudit
            top = contract.selected_doc_ids()[:1]
            audit = LargeAudit(sample_id=sample.sample_id, round=round_id,
                               final_answer="", used_doc_ids=top)

        # 3) 规则验证
        verification = verify(sample, contract, audit, json_valid=json_valid)

        # 4) reward（消融：use_decomposed_reward=False → 退回 answer-only reward）
        if cfg.ablation.use_decomposed_reward:
            reward = compute_decomposed_reward(sample, contract, audit, verification, self.cfg.reward)
        else:
            from ..schemas import RewardBreakdown
            ar = 1.0 if verification.answer_match else 0.0
            reward = RewardBreakdown(answer_reward=ar, total_reward=ar)
        targets = build_training_targets(sample, contract, audit, verification, reward)

        return ReplayExperience(
            sample_id=sample.sample_id, round=round_id,
            question=sample.question, answers=sample.answers,
            documents=sample.documents,
            contract=contract.to_dict(), audit=audit.to_dict(),
            verification=verification.to_dict(), rewards=reward.to_dict(),
            training_targets=targets)

    # --------------------------------------------------------- 一整轮
    def run_round(self, samples, round_id: int) -> dict:
        experiences = [self.make_experience(s, round_id) for s in samples]
        n = self.replay.write(experiences, round_id)
        self._write_jsonl(
            os.path.join(self.cfg.output_dir, "contracts", f"round_{round_id:03d}.jsonl"),
            [e.contract for e in experiences],
        )
        self._write_jsonl(
            os.path.join(self.cfg.output_dir, "audits", f"round_{round_id:03d}.jsonl"),
            [e.audit for e in experiences],
        )

        stats = {"round": round_id, "num_experiences": n}

        # 训练小模型
        if self.cfg.ablation.train_small_lora and self.small_trainer is not None:
            exps = self.replay.read(round_id)
            exps = self.replay.downsample_noisy(exps)
            pairs = self.replay.sample_small_training_pairs(exps)
            stats["small"] = self.small_trainer.train(pairs)
            small_dir = checkpoint_round_dir(self.cfg.models.small_lora_dir, round_id)
            os.makedirs(small_dir, exist_ok=True)
            self.small_trainer.save(small_dir)
            stats["small_checkpoint"] = small_dir

        # 训练大模型
        if self.cfg.ablation.train_large_lora and self.large_trainer is not None:
            exps = self.replay.read(round_id)
            sft = self.replay.sample_large_sft(exps)
            stats["large"] = self.large_trainer.train_sft(sft)
            large_dir = checkpoint_round_dir(self.cfg.models.large_lora_dir, round_id)
            os.makedirs(large_dir, exist_ok=True)
            self.large_trainer.save(large_dir)
            stats["large_checkpoint"] = large_dir

        # 评估
        if self.evaluator is not None:
            stats["eval"] = self.evaluator.evaluate(round_id)

        with open(os.path.join(self.cfg.output_dir, "metrics", f"round_{round_id:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        write_weight_manifest(self.cfg, self.cfg.output_dir)
        return stats

    def run(self, samples, start_round: int = 0) -> list[dict]:
        all_stats = []
        for round_id in range(start_round, self.cfg.training.num_rounds):
            all_stats.append(self.run_round(samples, round_id))
        return all_stats
