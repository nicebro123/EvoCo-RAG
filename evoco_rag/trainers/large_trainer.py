"""Large model LoRA training.

The mainline trains the large generator/auditor with supervised JSON audit
targets built from verifier-backed evidence.
"""

from __future__ import annotations

import json
from typing import Optional

from ..auditor import build_audit_prompt


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

    def save(self, out_dir: str) -> None:
        self.auditor.save_lora(out_dir)
