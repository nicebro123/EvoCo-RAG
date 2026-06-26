"""Large model LoRA / GRPO training.

The mainline keeps the small model as a reranker.  For the large
generator/auditor, this trainer supports:

  - SFT: supervised JSON audit targets built from verifier-backed evidence.
  - GRPO: CoRAG-style grouped sampling with rewards computed by our
    verifier/decomposed-reward stack, rather than answer-only matching.
"""

from __future__ import annotations

import json
import tempfile
from typing import Optional

from ..auditor import build_audit_prompt, parse_audit
from ..rewards import RewardWeights, compute_decomposed_reward
from ..schemas import EvidenceContract, RagSample, RewardBreakdown
from ..verifier import verify


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

    def __init__(self, auditor, lr: float = 1e-5, max_prompt_length: int = 3072,
                 max_completion_length: int = 1024, batch_size: int = 2):
        self.auditor = auditor
        self.lr = lr
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.batch_size = max(1, int(batch_size))

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
        messages = build_audit_prompt(sample, contract, show_gold=False)
        payload = exp.training_targets.get("large_sft_target")
        if not isinstance(payload, dict):
            payload = {key: exp.audit.get(key) for key in self.TARGET_FIELDS}
        if not str(payload.get("final_answer") or "").strip():
            return None
        target_payload = {key: payload.get(key) for key in self.TARGET_FIELDS}
        target = json.dumps(target_payload, ensure_ascii=False, separators=(",", ":"))
        return {"messages": messages, "target": target}

    def train_sft(self, experiences: list, epochs: int = 1, batch_size: int | None = None) -> dict:
        import torch
        from torch.optim import Adam

        model = self.auditor.model
        tok = self.auditor.tokenizer
        effective_batch_size = max(1, int(batch_size or self.batch_size))

        examples = [e for e in (self._build_sft_example(x) for x in experiences) if e]
        if not examples:
            return {
                "method": "sft",
                "trained_samples": 0,
                "avg_loss": None,
                "batch_size": effective_batch_size,
                "steps": 0,
            }

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        model.train()
        total_loss, steps = 0.0, 0
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
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    steps += 1
        finally:
            tok.padding_side = old_padding_side

        return {
            "method": "sft",
            "trained_samples": len(examples),
            "avg_loss": (total_loss / steps) if steps else None,
            "batch_size": effective_batch_size,
            "steps": steps,
        }

    # ---------------------------------------------------------------- GRPO
    @staticmethod
    def _completion_text(completion) -> str:
        """Normalize TRL completion objects across standard/chat formats."""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            first = completion[0]
            if isinstance(first, dict):
                return str(first.get("content") or "")
        if isinstance(completion, dict):
            return str(completion.get("content") or "")
        return str(completion or "")

    @staticmethod
    def make_grpo_reward_func(
        reward_weights: RewardWeights | None = None,
        use_decomposed_reward: bool = True,
        reward_log: list[float] | None = None,
    ):
        """Build a TRL-compatible reward function for generated audit JSON.

        The prompt never contains gold answers.  Gold answers/documents are
        carried only in hidden dataset columns for reward computation, matching
        the usual RL training setup.
        """
        weights = reward_weights or RewardWeights()

        def reward_func(completions, sample_json=None, contract_json=None, round_id=None, **kwargs):
            rewards: list[float] = []
            sample_json = sample_json or []
            contract_json = contract_json or []
            round_id = round_id or []
            for i, completion in enumerate(completions):
                try:
                    sample_payload = json.loads(sample_json[i])
                    contract_payload = json.loads(contract_json[i])
                    sample = RagSample.from_dict(sample_payload)
                    contract = EvidenceContract.from_dict(contract_payload)
                    rid = int(round_id[i]) if i < len(round_id) else contract.round
                    audit, json_valid = parse_audit(
                        LargeTrainer._completion_text(completion),
                        sample.sample_id,
                        rid,
                    )
                    verification = verify(sample, contract, audit, json_valid=json_valid)
                    if use_decomposed_reward:
                        reward = compute_decomposed_reward(
                            sample, contract, audit, verification, weights)
                    else:
                        reward = RewardBreakdown(
                            answer_reward=1.0 if verification.answer_match else 0.0,
                            total_reward=1.0 if verification.answer_match else 0.0,
                        )
                    value = float(reward.total_reward)
                except Exception:
                    # Invalid metadata or unparsable output should be a bad
                    # sample, not a crashed training run.
                    value = -1.0
                rewards.append(value)
            if reward_log is not None:
                reward_log.extend(rewards)
            return rewards

        return reward_func

    def _build_grpo_example(self, exp) -> Optional[dict]:
        """Convert a replay experience into one GRPO prompt row."""
        if not exp.contract or not exp.documents:
            return None
        sample = RagSample(
            sample_id=exp.sample_id,
            question=exp.question,
            answers=exp.answers,
            documents=exp.documents,
        )
        contract = EvidenceContract.from_dict(exp.contract)
        messages = build_audit_prompt(sample, contract, show_gold=False)
        prompt = self.auditor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "sample_id": exp.sample_id,
            "round_id": int(exp.round),
            "sample_json": json.dumps(sample.to_dict(), ensure_ascii=False),
            "contract_json": json.dumps(contract.to_dict(), ensure_ascii=False),
        }

    def train_grpo(
        self,
        experiences: list,
        reward_weights: RewardWeights | None = None,
        use_decomposed_reward: bool = True,
        num_generations: int = 2,
        n_per_train: int = 1,
        epochs: float = 1.0,
        beta: float = 0.04,
        temperature: float = 0.7,
        gradient_accumulation_steps: int = 1,
        max_steps: int | None = None,
        output_dir: str | None = None,
    ) -> dict:
        """Train the large model with verifier-backed GRPO.

        This is CoRAG-style in the important sense: each prompt samples a group
        of completions and optimizes relative rewards. The reward itself is
        anchored by deterministic normalized EM/sub-string checks over the
        generated audit JSON, not by LLM-as-judge.
        """
        num_generations = max(2, int(num_generations))
        n_per_train = max(1, int(n_per_train))
        effective_batch_size = num_generations * n_per_train

        examples = [e for e in (self._build_grpo_example(x) for x in experiences) if e]
        if not examples:
            return {
                "method": "grpo",
                "trained_samples": 0,
                "avg_reward": None,
                "num_generations": num_generations,
                "n_per_train": n_per_train,
                "batch_size": effective_batch_size,
                "steps": 0,
            }

        try:
            from datasets import Dataset
            from trl import GRPOConfig, GRPOTrainer
        except Exception as exc:  # pragma: no cover - depends on training env
            raise RuntimeError(
                "GRPO training requires `trl` and `datasets` in the active environment"
            ) from exc

        import torch

        model = self.auditor.model
        tok = self.auditor.tokenizer
        tok.padding_side = "left"
        dataset = Dataset.from_list(examples)
        rewards_seen: list[float] = []
        reward_func = self.make_grpo_reward_func(
            reward_weights=reward_weights,
            use_decomposed_reward=use_decomposed_reward,
            reward_log=rewards_seen,
        )
        args = GRPOConfig(
            output_dir=output_dir or tempfile.mkdtemp(prefix="evoco_grpo_"),
            save_strategy="no",
            report_to="none",
            num_train_epochs=float(epochs),
            max_steps=-1 if max_steps is None else int(max_steps),
            num_generations=num_generations,
            per_device_train_batch_size=effective_batch_size,
            gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
            learning_rate=self.lr,
            max_prompt_length=self.max_prompt_length,
            max_completion_length=self.max_completion_length,
            temperature=float(temperature),
            beta=float(beta),
            logging_steps=1,
            logging_first_step=True,
            bf16=bool(torch.cuda.is_available()),
            fp16=False,
            remove_unused_columns=False,
        )
        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            reward_funcs=reward_func,
            processing_class=tok,
        )
        output = trainer.train()
        steps = int(getattr(output, "global_step", 0) or 0)
        avg_reward = (sum(rewards_seen) / len(rewards_seen)) if rewards_seen else None
        return {
            "method": "grpo",
            "trained_samples": len(examples),
            "avg_reward": avg_reward,
            "num_generations": num_generations,
            "n_per_train": n_per_train,
            "batch_size": effective_batch_size,
            "steps": steps,
            "training_loss": getattr(output, "training_loss", None),
        }

    def save(self, out_dir: str) -> None:
        self.auditor.save_lora(out_dir)
