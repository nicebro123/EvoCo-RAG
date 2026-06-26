"""Small reranker LoRA training.

The mainline uses replay experiences to build positive / negative document
pairs and optimizes a margin-ranking loss.  No action head is trained here:
the small model is a reranker, while the large model handles generation and
audit.
"""

from __future__ import annotations

from typing import Optional


def _binary_ece(probs: list[float], labels: list[float], n_bins: int = 10) -> Optional[float]:
    if not probs:
        return None
    total = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idxs = [
            j for j, p in enumerate(probs)
            if (lo <= p <= hi if i == 0 else lo < p <= hi)
        ]
        if not idxs:
            continue
        conf = sum(probs[j] for j in idxs) / len(idxs)
        acc = sum(labels[j] for j in idxs) / len(idxs)
        ece += (len(idxs) / total) * abs(acc - conf)
    return float(ece)


class SmallTrainer:
    def __init__(self, policy, lr: float = 5e-5, margin: float = 1.0,
                 max_length: int = 512, batch_size: int = 4,
                 evidence_loss_weight: float = 1.0):
        import torch  # noqa: F401
        self.policy = policy
        self.lr = lr
        self.margin = margin
        self.max_length = max_length
        self.batch_size = batch_size
        self.evidence_loss_weight = evidence_loss_weight

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
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = Adam(params, lr=self.lr)

        usable = [
            p for p in pairs
            if p.get("documents")
            and (p.get("positive_doc_ids") or p.get("negative_doc_ids"))
        ]
        if not usable:
            return {
                "trained_samples": 0,
                "skipped": len(pairs),
                "steps": 0,
                "avg_loss": None,
            }

        model.train()
        total_loss, steps = 0.0, 0
        rank_loss_total = 0.0
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
                out = model(
                    **inputs,
                    return_dict=True,
                )
                scores = out.logits.view(-1)

                batch_loss = scores.sum() * 0.0
                rank_valid = 0
                for i in range(len(pos_lens)):
                    s, e = split[i], split[i + 1]
                    pl = pos_lens[i]
                    if pl == 0 or (e - s - pl) == 0:
                        continue
                    pos_scores = scores[s:s + pl]
                    neg_scores = scores[s + pl:e]
                    for ps in pos_scores:
                        batch_loss = batch_loss + torch.sum(F.relu(neg_scores - ps + self.margin))
                    rank_valid += 1
                rank_loss = batch_loss / rank_valid if rank_valid else scores.sum() * 0.0
                loss = rank_loss
                if rank_valid == 0:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                rank_loss_total += float(rank_loss.item())
                steps += 1

        result = {
            "trained_samples": len(usable),
            "skipped": len(pairs) - len(usable),
            "steps": steps,
            "avg_loss": (total_loss / steps) if steps else None,
            "avg_rank_loss": (rank_loss_total / steps) if steps else None,
        }
        return result

    def save(self, out_dir: str) -> None:
        self.policy.save_lora(out_dir)
