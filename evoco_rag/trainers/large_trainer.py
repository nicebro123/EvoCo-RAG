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


class LargeTrainer:
    def __init__(self, auditor, lr: float = 1e-5, max_prompt_length: int = 2048,
                 max_completion_length: int = 1024):
        self.auditor = auditor
        self.lr = lr
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

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
        # 评估阶段不可见 gold；SFT 目标就是已审计通过的 audit JSON
        messages = build_audit_prompt(sample, contract, show_gold=False)
        target = json.dumps(exp.audit, ensure_ascii=False)
        return {"messages": messages, "target": target}

    def train_sft(self, experiences: list, epochs: int = 1, batch_size: int = 2) -> dict:
        import torch
        from torch.optim import Adam

        model = self.auditor.model
        tok = self.auditor.tokenizer

        examples = [e for e in (self._build_sft_example(x) for x in experiences) if e]
        if not examples:
            return {"trained_samples": 0, "avg_loss": None}

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        model.train()
        total_loss, steps = 0.0, 0
        for _ in range(epochs):
            for ex in examples:
                prompt = tok.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=True)
                full = prompt + ex["target"] + tok.eos_token
                enc = tok(full, return_tensors="pt", truncation=True,
                          max_length=self.max_prompt_length + self.max_completion_length)
                input_ids = enc["input_ids"].to(model.device)
                # 只对 target 部分计算 loss
                prompt_len = tok(prompt, return_tensors="pt",
                                 truncation=True, max_length=self.max_prompt_length)["input_ids"].shape[1]
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                out = model(input_ids=input_ids, attention_mask=enc["attention_mask"].to(model.device),
                            labels=labels)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                steps += 1

        return {"trained_samples": len(examples), "avg_loss": (total_loss / steps) if steps else None}

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
