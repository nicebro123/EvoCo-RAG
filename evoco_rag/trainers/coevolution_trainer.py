"""协同进化主循环（开发文档 §5.12）。

调度一整轮：小模型出合约 → 大模型审计 → 规则验证 → 分解 reward → 写 replay
→ 训练小模型 LoRA → 训练大模型 LoRA → 评估 → 存 checkpoint。
支持消融开关（ablation）与 resume。
"""

from __future__ import annotations

import json
import os
import time

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

    @staticmethod
    def _write_jsonl_record(handle, record: dict) -> None:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h{m:02d}m{s:02d}s"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    def _progress_interval(self) -> int:
        return max(0, int(getattr(self.cfg.runtime, "progress_interval", 50)))

    def _replay_flush_interval(self) -> int:
        return max(0, int(getattr(self.cfg.runtime, "replay_flush_interval", 10)))

    def _log_experience_progress(
        self,
        round_id: int,
        done: int,
        total: int,
        generated: int,
        started_at: float,
    ) -> None:
        elapsed = time.time() - started_at
        rate = generated / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - done)
        eta = (remaining / rate) if rate > 0 else 0.0
        print(
            f"round {round_id}: experience {done}/{total} "
            f"elapsed={self._format_seconds(elapsed)} "
            f"rate={rate:.2f}/s eta={self._format_seconds(eta)}",
            flush=True,
        )

    def _valid_partial_experiences(self, round_id: int, sample_ids: set[str]) -> list[ReplayExperience]:
        existing = []
        seen = set()
        skipped = 0
        path = self.replay.round_path(round_id)
        if not os.path.exists(path):
            return existing
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = ReplayExperience.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    skipped += 1
                    continue
                if exp.sample_id in sample_ids and exp.sample_id not in seen:
                    existing.append(exp)
                    seen.add(exp.sample_id)
        if skipped:
            print(
                f"round {round_id}: skipped {skipped} invalid partial replay lines",
                flush=True,
            )
        return existing

    def _rewrite_round_artifacts(self, round_id: int, experiences: list[ReplayExperience]) -> None:
        replay_path = self.replay.round_path(round_id)
        contracts_path = os.path.join(
            self.cfg.output_dir, "contracts", f"round_{round_id:03d}.jsonl")
        audits_path = os.path.join(
            self.cfg.output_dir, "audits", f"round_{round_id:03d}.jsonl")
        self._write_jsonl(replay_path, [e.to_dict() for e in experiences])
        self._write_jsonl(contracts_path, [e.contract for e in experiences])
        self._write_jsonl(audits_path, [e.audit for e in experiences])

    # --------------------------------------------------- 单样本经验生成
    def make_experience(self, sample, round_id: int) -> ReplayExperience:
        cfg = self.cfg
        # 1) 小模型证据合约
        if self.small is not None:
            contract = self.small.build_contract(
                sample, round_id=round_id, top_k=cfg.contract.top_k,
                high_conf_threshold=cfg.contract.high_conf_threshold,
                answer_now_margin=cfg.contract.answer_now_margin,
                max_selected_docs=cfg.contract.max_selected_docs,
                action_mode=cfg.contract.action_mode,
                policy_action_min_conf=cfg.contract.policy_action_min_conf)
        else:
            # 无模型的纯逻辑回放（测试/调试用）：按文档原序当作打分
            ranked = [{"doc_id": d["doc_id"], "score": -i}
                      for i, d in enumerate(sample.documents)]
            contract = build_contract(sample, ranked, round_id=round_id,
                                      top_k=cfg.contract.top_k,
                                      high_conf_threshold=cfg.contract.high_conf_threshold,
                                      answer_now_margin=cfg.contract.answer_now_margin,
                                      max_selected_docs=cfg.contract.max_selected_docs,
                                      action_mode=cfg.contract.action_mode,
                                      policy_action_min_conf=cfg.contract.policy_action_min_conf)

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
        round_started = time.time()
        total = len(samples)
        sample_ids = {s.sample_id for s in samples}
        replay_path = self.replay.round_path(round_id)
        contracts_path = os.path.join(
            self.cfg.output_dir, "contracts", f"round_{round_id:03d}.jsonl")
        audits_path = os.path.join(
            self.cfg.output_dir, "audits", f"round_{round_id:03d}.jsonl")
        for path in (replay_path, contracts_path, audits_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        existing = self._valid_partial_experiences(round_id, sample_ids)
        done_ids = {e.sample_id for e in existing}
        if existing:
            self._rewrite_round_artifacts(round_id, existing)
            self.replay.rebuild_all()
            print(
                f"round {round_id}: resumed {len(existing)}/{total} existing experiences",
                flush=True,
            )
        pending = [sample for sample in samples if sample.sample_id not in done_ids]

        print(
            f"round {round_id}: generating {len(pending)} remaining experiences "
            f"for {total} samples",
            flush=True,
        )
        generation_started = time.time()
        progress_interval = self._progress_interval()
        flush_interval = self._replay_flush_interval()
        n = len(existing)
        if pending:
            mode = "a" if existing else "w"
            with open(replay_path, mode, encoding="utf-8") as fr, \
                    open(contracts_path, mode, encoding="utf-8") as fc, \
                    open(audits_path, mode, encoding="utf-8") as fa:
                for idx, sample in enumerate(pending, start=len(existing) + 1):
                    exp = self.make_experience(sample, round_id)
                    self._write_jsonl_record(fr, exp.to_dict())
                    self._write_jsonl_record(fc, exp.contract)
                    self._write_jsonl_record(fa, exp.audit)
                    n += 1

                    if flush_interval and (idx % flush_interval == 0):
                        fr.flush()
                        fc.flush()
                        fa.flush()
                    if progress_interval and (idx % progress_interval == 0 or idx == total):
                        generated = idx - len(existing)
                        self._log_experience_progress(
                            round_id, idx, total, generated, generation_started)
        elif not existing:
            with open(replay_path, "w", encoding="utf-8"):
                pass
            with open(contracts_path, "w", encoding="utf-8"):
                pass
            with open(audits_path, "w", encoding="utf-8"):
                pass

        self.replay.rebuild_all()
        generation_seconds = time.time() - generation_started
        print(
            f"round {round_id}: wrote {n} experiences -> {replay_path}",
            flush=True,
        )

        stats = {
            "round": round_id,
            "num_experiences": n,
            "resumed_experiences": len(existing),
            "generated_experiences": len(pending),
            "timing": {
                "experience_generation_seconds": round(generation_seconds, 4),
            },
        }

        # 训练小模型
        small_started = time.time()
        if self.cfg.ablation.train_small_lora and self.small_trainer is not None:
            exps = self.replay.read(round_id)
            exps = self.replay.downsample_noisy(exps)
            pairs = self.replay.sample_small_training_pairs(exps)
            stats["small"] = self.small_trainer.train(pairs)
            small_dir = checkpoint_round_dir(self.cfg.models.small_lora_dir, round_id)
            os.makedirs(small_dir, exist_ok=True)
            self.small_trainer.save(small_dir)
            stats["small_checkpoint"] = small_dir
        stats["timing"]["small_training_seconds"] = round(time.time() - small_started, 4)
        print(
            f"round {round_id}: small training "
            f"{self._format_seconds(stats['timing']['small_training_seconds'])}",
            flush=True,
        )

        # 训练大模型
        large_started = time.time()
        if self.cfg.ablation.train_large_lora and self.large_trainer is not None:
            exps = self.replay.read(round_id)
            sft = self.replay.sample_large_sft(exps)
            stats["large"] = self.large_trainer.train_sft(sft)
            large_dir = checkpoint_round_dir(self.cfg.models.large_lora_dir, round_id)
            os.makedirs(large_dir, exist_ok=True)
            self.large_trainer.save(large_dir)
            stats["large_checkpoint"] = large_dir
        stats["timing"]["large_training_seconds"] = round(time.time() - large_started, 4)
        print(
            f"round {round_id}: large training "
            f"{self._format_seconds(stats['timing']['large_training_seconds'])}",
            flush=True,
        )

        # 评估
        eval_started = time.time()
        if self.evaluator is not None:
            stats["eval"] = self.evaluator.evaluate(round_id)
        stats["timing"]["evaluation_seconds"] = round(time.time() - eval_started, 4)
        stats["timing"]["total_round_seconds"] = round(time.time() - round_started, 4)
        print(
            f"round {round_id}: evaluation "
            f"{self._format_seconds(stats['timing']['evaluation_seconds'])}; "
            f"total {self._format_seconds(stats['timing']['total_round_seconds'])}",
            flush=True,
        )

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
