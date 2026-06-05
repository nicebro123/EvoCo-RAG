"""小模型 LoRA 训练（开发文档 §5.10、§9.3）。

用 replay buffer 里经审计的 positive / negative doc 训练 reranker，沿用 margin
ranking loss。第一版只做文档级 ranking；evidence/action/calibration 多任务 loss
留待后续阶段扩展。torch 延迟导入。
"""

from __future__ import annotations

from typing import Optional


class SmallTrainer:
    def __init__(self, policy, lr: float = 5e-5, margin: float = 1.0,
                 max_length: int = 512, batch_size: int = 4):
        import torch  # noqa: F401
        self.policy = policy
        self.lr = lr
        self.margin = margin
        self.max_length = max_length
        self.batch_size = batch_size

    def _doc_text(self, documents: list[dict], doc_id: int) -> str:
        for d in documents:
            if d.get("doc_id") == doc_id:
                return d.get("text") or d.get("raw") or ""
        return ""

    def train(self, pairs: list[dict], epochs: int = 1) -> dict:
        """pairs 来自 ReplayBuffer.sample_small_training_pairs。

        每条含 positive_doc_ids / negative_doc_ids / documents / question。
        """
        import torch
        import torch.nn.functional as F
        from torch.optim import Adam

        model = self.policy.model
        tok = self.policy.tokenizer
        device = self.policy.device
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

        usable = [p for p in pairs if p["positive_doc_ids"] and p["negative_doc_ids"]]
        if not usable:
            return {"trained_samples": 0, "skipped": len(pairs), "avg_loss": None}

        model.train()
        total_loss, steps = 0.0, 0
        for _ in range(epochs):
            for start in range(0, len(usable), self.batch_size):
                batch = usable[start:start + self.batch_size]
                all_pairs, split, pos_lens = [], [0], []
                for ex in batch:
                    pos = [(ex["question"], self._doc_text(ex["documents"], d))
                           for d in ex["positive_doc_ids"]]
                    neg = [(ex["question"], self._doc_text(ex["documents"], d))
                           for d in ex["negative_doc_ids"]]
                    all_pairs.extend(pos + neg)
                    split.append(len(all_pairs))
                    pos_lens.append(len(pos))

                inputs = tok(all_pairs, padding=True, truncation=True,
                             return_tensors="pt", max_length=self.max_length).to(device)
                scores = model(**inputs, return_dict=True).logits.view(-1)

                batch_loss = 0.0
                valid = 0
                for i in range(len(pos_lens)):
                    s, e = split[i], split[i + 1]
                    pl = pos_lens[i]
                    if pl == 0 or (e - s - pl) == 0:
                        continue
                    pos_scores = scores[s:s + pl]
                    neg_scores = scores[s + pl:e]
                    for ps in pos_scores:
                        batch_loss = batch_loss + torch.sum(F.relu(neg_scores - ps + self.margin))
                    valid += 1
                if valid == 0:
                    continue
                batch_loss = batch_loss / valid

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_loss += float(batch_loss.item())
                steps += 1

        return {
            "trained_samples": len(usable),
            "skipped": len(pairs) - len(usable),
            "avg_loss": (total_loss / steps) if steps else None,
        }

    def save(self, out_dir: str) -> None:
        self.policy.save_lora(out_dir)
