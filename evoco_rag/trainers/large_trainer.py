"""大模型 LoRA 训练（开发文档 §5.11、§9.1、§9.2）。

两种模式：
  - SFT  : 用 replay buffer 中 large_sft_eligible=true 的高质量样本，学习
           "结构化答案 + 审计 JSON" 的目标输出。
  - GRPO : 复用 TRL GRPOTrainer，但 reward_funcs 改为读取结构化 audit/verification
           计算 decomposed reward（不再 answer-only，也不在 reward 里写训练文件）。
torch / trl 延迟导入。
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from ..auditor import build_audit_prompt
from ..cabl import build_boundary_pairs, build_relation_answer_pool


class LargeTrainer:
    TARGET_FIELDS = (
        "final_answer",
        "used_doc_ids",
        "used_evidence",
        "answer_correctness",
        "support_level",
        "failure_type",
        "small_model_feedback",
        "suggested_action",
    )

    def __init__(
        self,
        auditor,
        lr: float = 1e-5,
        max_prompt_length: int = 3072,
        max_completion_length: int = 1024,
        batch_size: int = 2,
        cabl_enabled: bool = False,
        cabl_loss_weight: float = 0.2,
        cabl_margin: float = 0.5,
        cabl_max_negatives_per_sample: int = 3,
        cabl_min_negative_chars: int = 2,
        cabl_max_prompt_length: int = 768,
        cabl_evidence_char_limit: int = 512,
        cabl_use_model_self_error: bool = True,
        cabl_use_relation_answer_pool: bool = False,
        cabl_use_answer_type_filter: bool = False,
        cabl_use_retrieved_distractors: bool = True,
        cabl_use_counterfactual_evidence: bool = False,
        cabl_hard_aware_enabled: bool = False,
        cabl_hard_pair_weight: float = 2.0,
        cabl_skip_retrieval_absent: bool = True,
        cabl_relation_hint_enabled: bool = True,
    ):
        self.auditor = auditor
        self.lr = lr
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.batch_size = max(1, int(batch_size))
        self.cabl_enabled = bool(cabl_enabled)
        self.cabl_loss_weight = float(cabl_loss_weight)
        self.cabl_margin = float(cabl_margin)
        self.cabl_max_negatives_per_sample = max(0, int(cabl_max_negatives_per_sample))
        self.cabl_min_negative_chars = max(1, int(cabl_min_negative_chars))
        self.cabl_max_prompt_length = max(64, int(cabl_max_prompt_length))
        self.cabl_evidence_char_limit = max(0, int(cabl_evidence_char_limit))
        self.cabl_use_model_self_error = bool(cabl_use_model_self_error)
        self.cabl_use_relation_answer_pool = bool(cabl_use_relation_answer_pool)
        self.cabl_use_answer_type_filter = bool(cabl_use_answer_type_filter)
        self.cabl_use_retrieved_distractors = bool(cabl_use_retrieved_distractors)
        self.cabl_use_counterfactual_evidence = bool(cabl_use_counterfactual_evidence)
        self.cabl_hard_aware_enabled = bool(cabl_hard_aware_enabled)
        self.cabl_hard_pair_weight = float(cabl_hard_pair_weight)
        self.cabl_skip_retrieval_absent = bool(cabl_skip_retrieval_absent)
        self.cabl_relation_hint_enabled = bool(cabl_relation_hint_enabled)
        self._cabl_answer_pool = None

    # ----------------------------------------------------------------- SFT
    def _build_sft_example(self, exp) -> Optional[dict]:
        """把一条经验转成 (prompt, target_json) SFT 样本。"""
        from ..schemas import EvidenceContract, RagSample

        sample = RagSample(
            sample_id=exp.sample_id, question=exp.question,
            answers=exp.answers, documents=exp.documents)
        contract = EvidenceContract.from_dict(exp.contract) if exp.contract else None
        if contract is None:
            return None
        # Prompt never contains gold. The target is a compact, verifier-built
        # correction and deliberately excludes audit metadata/raw generations.
        messages = build_audit_prompt(
            sample,
            contract,
            show_gold=False,
            candidate_doc_char_limit=getattr(self.auditor, "candidate_doc_char_limit", 1200),
        )
        payload = exp.training_targets.get("large_sft_target")
        if not isinstance(payload, dict):
            payload = {key: exp.audit.get(key) for key in self.TARGET_FIELDS}
        if not str(payload.get("final_answer") or "").strip():
            return None
        target_payload = {key: payload.get(key) for key in self.TARGET_FIELDS}
        target = json.dumps(target_payload, ensure_ascii=False, separators=(",", ":"))
        boundary_pairs = []
        if self.cabl_enabled:
            boundary_pairs = build_boundary_pairs(
                exp,
                max_negatives=self.cabl_max_negatives_per_sample,
                margin=self.cabl_margin,
                min_negative_chars=self.cabl_min_negative_chars,
                answer_pool=self._cabl_answer_pool,
                use_model_self_error=self.cabl_use_model_self_error,
                use_relation_answer_pool=self.cabl_use_relation_answer_pool,
                use_answer_type_filter=self.cabl_use_answer_type_filter,
                use_retrieved_distractors=self.cabl_use_retrieved_distractors,
                use_counterfactual_evidence=self.cabl_use_counterfactual_evidence,
                hard_aware=self.cabl_hard_aware_enabled,
                hard_pair_weight=self.cabl_hard_pair_weight,
                skip_retrieval_absent=self.cabl_skip_retrieval_absent,
            )
        return {
            "messages": messages,
            "target": target,
            "boundary_pairs": boundary_pairs,
            "weight": max(0.0, float(exp.training_targets.get("large_sft_weight", 1.0))),
            "target_source": exp.training_targets.get("large_sft_target_source"),
        }

    def _target_logprobs(
        self,
        prompts: list[str],
        targets: list[str],
        *,
        max_prompt_length: int | None = None,
        max_completion_length: int = 64,
    ):
        """Return mean completion log-probabilities for prompt+target strings.

        This helper is used by CABL. It deliberately scores short answer strings
        rather than full JSON targets, because CABL is about separating factual
        answer candidates, not about formatting.
        """

        import torch
        import torch.nn.functional as F

        if not prompts:
            return None
        model = self.auditor.model
        tok = self.auditor.tokenizer
        fulls = [prompt + str(target) + tok.eos_token for prompt, target in zip(prompts, targets)]
        prompt_limit = int(max_prompt_length or self.max_prompt_length)
        enc = tok(
            fulls,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_limit + int(max_completion_length),
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for row, prompt in enumerate(prompts):
            prompt_len = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=prompt_limit,
            )["input_ids"].shape[1]
            nonpad_len = int(attention_mask[row].sum().item())
            labels[row, :min(prompt_len, nonpad_len)] = -100
            if prompt_len >= nonpad_len:
                labels[row, :] = -100
        if not torch.any(labels != -100):
            return None
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = getattr(out, "logits", None)
        if logits is None:
            return None
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_logprobs = F.log_softmax(shift_logits, dim=-1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs = token_logprobs * valid
        lengths = valid.sum(dim=1).clamp_min(1)
        return token_logprobs.sum(dim=1) / lengths

    def _boundary_prompt(self, pair: dict) -> str:
        evidence = str(pair.get("evidence") or "").strip()
        if self.cabl_evidence_char_limit and len(evidence) > self.cabl_evidence_char_limit:
            evidence = evidence[: self.cabl_evidence_char_limit].rstrip() + " ..."
        parts = [
            "Choose the most factually correct short answer.",
            f"Question: {pair.get('question') or ''}",
        ]
        relation = str(pair.get("relation") or "").strip()
        relation_hint = str(pair.get("relation_hint") or "").strip()
        if self.cabl_relation_hint_enabled and relation:
            parts.append(f"Question relation: {relation}")
        if self.cabl_relation_hint_enabled and relation_hint:
            parts.append(f"Relation constraint: {relation_hint}")
        if evidence:
            parts.append(f"Evidence: {evidence}")
        counterfactual_evidence = str(pair.get("counterfactual_evidence") or "").strip()
        if counterfactual_evidence:
            if self.cabl_evidence_char_limit and len(counterfactual_evidence) > self.cabl_evidence_char_limit:
                counterfactual_evidence = counterfactual_evidence[: self.cabl_evidence_char_limit].rstrip() + " ..."
            parts.append(f"Counterfactual distractor evidence: {counterfactual_evidence}")
        parts.append("Answer: ")
        return "\n".join(parts)

    def _boundary_loss(self, examples: list[dict]):
        import torch
        import torch.nn.functional as F

        pair_prompts, positives, negatives, margins, weights = [], [], [], [], []
        for ex in examples:
            for pair in ex.get("boundary_pairs", []) or []:
                pos = str(pair.get("positive") or "").strip()
                neg = str(pair.get("negative") or "").strip()
                if not pos or not neg:
                    continue
                pair_prompts.append(self._boundary_prompt(pair))
                positives.append(pos)
                negatives.append(neg)
                margins.append(float(pair.get("margin", self.cabl_margin)))
                weights.append(max(0.0, float(pair.get("weight", 1.0))))
        if not pair_prompts:
            return None, 0
        all_prompts = pair_prompts + pair_prompts
        all_targets = positives + negatives
        logprobs = self._target_logprobs(
            all_prompts,
            all_targets,
            max_prompt_length=self.cabl_max_prompt_length,
            max_completion_length=64,
        )
        if logprobs is None:
            return None, len(pair_prompts)
        n = len(pair_prompts)
        pos_lp = logprobs[:n]
        neg_lp = logprobs[n:]
        margin = torch.tensor(margins, dtype=pos_lp.dtype, device=pos_lp.device)
        weight = torch.tensor(weights, dtype=pos_lp.dtype, device=pos_lp.device)
        raw_loss = F.relu(margin - pos_lp + neg_lp)
        normalizer = weight.sum().clamp_min(1.0)
        return (raw_loss * weight).sum() / normalizer, n

    def train_sft(self, experiences: list, epochs: int = 1, batch_size: int | None = None) -> dict:
        import torch
        from torch.optim import Adam

        model = self.auditor.model
        tok = self.auditor.tokenizer
        effective_batch_size = max(1, int(batch_size or self.batch_size))

        if self.cabl_enabled and self.cabl_use_relation_answer_pool:
            self._cabl_answer_pool = build_relation_answer_pool(experiences)
        else:
            self._cabl_answer_pool = None

        examples = [e for e in (self._build_sft_example(x) for x in experiences) if e]
        if not examples:
            return {
                "trained_samples": 0,
                "avg_loss": None,
                "avg_sft_loss": None,
                "avg_boundary_loss": None,
                "batch_size": effective_batch_size,
                "steps": 0,
                "boundary_pairs": 0,
                "cabl_enabled": self.cabl_enabled,
                "cabl_modules": {
                    "use_model_self_error": self.cabl_use_model_self_error,
                    "use_relation_answer_pool": self.cabl_use_relation_answer_pool,
                    "use_answer_type_filter": self.cabl_use_answer_type_filter,
                    "use_retrieved_distractors": self.cabl_use_retrieved_distractors,
                    "use_counterfactual_evidence": self.cabl_use_counterfactual_evidence,
                    "hard_aware_enabled": self.cabl_hard_aware_enabled,
                    "hard_pair_weight": self.cabl_hard_pair_weight,
                    "skip_retrieval_absent": self.cabl_skip_retrieval_absent,
                    "relation_hint_enabled": self.cabl_relation_hint_enabled,
                },
            }

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        model.train()
        total_loss, sft_loss_total, boundary_loss_total, steps = 0.0, 0.0, 0.0, 0
        boundary_steps, boundary_pairs_total = 0, 0
        old_padding_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "right"
        try:
            for _ in range(epochs):
                for start in range(0, len(examples), effective_batch_size):
                    batch = examples[start:start + effective_batch_size]
                    prompts = [
                        tok.apply_chat_template(
                            ex["messages"], tokenize=False, add_generation_prompt=True)
                        for ex in batch
                    ]
                    fulls = [prompt + ex["target"] + tok.eos_token
                             for prompt, ex in zip(prompts, batch)]
                    enc = tok(
                        fulls,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_prompt_length + self.max_completion_length,
                    )
                    input_ids = enc["input_ids"].to(model.device)
                    attention_mask = enc["attention_mask"].to(model.device)
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    for row, prompt in enumerate(prompts):
                        prompt_len = tok(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.max_prompt_length,
                        )["input_ids"].shape[1]
                        nonpad_len = int(attention_mask[row].sum().item())
                        labels[row, :min(prompt_len, nonpad_len)] = -100
                        if prompt_len >= nonpad_len:
                            labels[row, :] = -100
                    if not torch.any(labels != -100):
                        continue
                    weights = torch.tensor(
                        [max(0.0, float(ex.get("weight", 1.0))) for ex in batch],
                        dtype=torch.float32,
                        device=model.device,
                    )
                    if not torch.any(weights > 0):
                        continue
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    sft_loss = out.loss * weights.mean()
                    boundary_loss = None
                    boundary_pair_count = 0
                    optimizer.zero_grad()
                    sft_loss.backward()
                    loss_value = float(sft_loss.item())
                    if self.cabl_enabled and self.cabl_loss_weight > 0:
                        boundary_loss, boundary_pair_count = self._boundary_loss(batch)
                        boundary_pairs_total += boundary_pair_count
                        if boundary_loss is not None:
                            weighted_boundary = self.cabl_loss_weight * boundary_loss
                            weighted_boundary.backward()
                            loss_value += float(weighted_boundary.item())
                            boundary_loss_total += float(boundary_loss.item())
                            boundary_steps += 1
                    optimizer.step()
                    total_loss += loss_value
                    sft_loss_total += float(sft_loss.item())
                    steps += 1
        finally:
            tok.padding_side = old_padding_side

        return {
            "trained_samples": len(examples),
            "avg_loss": (total_loss / steps) if steps else None,
            "avg_sft_loss": (sft_loss_total / steps) if steps else None,
            "avg_boundary_loss": (
                boundary_loss_total / boundary_steps if boundary_steps else None
            ),
            "batch_size": effective_batch_size,
            "steps": steps,
            "boundary_pairs": boundary_pairs_total,
            "avg_sample_weight": (
                sum(float(ex.get("weight", 1.0)) for ex in examples) / len(examples)
            ) if examples else None,
            "target_source_distribution": {
                source: sum(
                    1 for ex in examples
                    if str(ex.get("target_source") or "unknown") == source
                )
                for source in sorted(
                    {str(ex.get("target_source") or "unknown") for ex in examples}
                )
            },
            "cabl_enabled": self.cabl_enabled,
            "cabl_modules": {
                "use_model_self_error": self.cabl_use_model_self_error,
                "use_relation_answer_pool": self.cabl_use_relation_answer_pool,
                "use_answer_type_filter": self.cabl_use_answer_type_filter,
                "use_retrieved_distractors": self.cabl_use_retrieved_distractors,
                "use_counterfactual_evidence": self.cabl_use_counterfactual_evidence,
                "hard_aware_enabled": self.cabl_hard_aware_enabled,
                "hard_pair_weight": self.cabl_hard_pair_weight,
                "skip_retrieval_absent": self.cabl_skip_retrieval_absent,
                "relation_hint_enabled": self.cabl_relation_hint_enabled,
            },
        }

    # ---------------------------------------------------------------- GRPO
    @staticmethod
    def make_grpo_reward_func(reward_lookup: Callable[[str], float]):
        """构造 TRL 兼容的 reward 函数：只返回大模型 reward，不写任何训练文件。

        reward_lookup: sample_id -> total_reward（由 replay/verification 预先算好）。
        """
        def reward_func(completions, **kwargs):
            sample_ids = kwargs.get("sample_id", [])
            rewards = []
            for i, _ in enumerate(completions):
                sid = sample_ids[i] if i < len(sample_ids) else None
                rewards.append(float(reward_lookup(sid)) if sid is not None else 0.0)
            return rewards
        return reward_func

    def save(self, out_dir: str) -> None:
        self.auditor.save_lora(out_dir)
