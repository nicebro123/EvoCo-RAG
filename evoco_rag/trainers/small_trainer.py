"""小模型 LoRA 训练（开发文档 §5.10、§9.3）。

用 replay buffer 里经审计的 positive / negative doc 训练 reranker，沿用 margin
ranking loss。开启 small policy heads 后，同时训练 evidence/action/calibration
多任务 loss，并返回 action/evidence/calibration 指标。torch 延迟导入。
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
                 evidence_loss_weight: float = 1.0,
                 action_loss_weight: float = 0.5,
                 calibration_loss_weight: float = 0.2,
                 score_pointwise_loss_weight: float = 0.0,
                 action_class_weights: Optional[list[float]] = None):
        import torch  # noqa: F401
        self.policy = policy
        self.lr = lr
        self.margin = margin
        self.max_length = max_length
        self.batch_size = batch_size
        self.evidence_loss_weight = evidence_loss_weight
        self.action_loss_weight = action_loss_weight
        self.calibration_loss_weight = calibration_loss_weight
        self.score_pointwise_loss_weight = float(score_pointwise_loss_weight)
        self.action_class_weights = action_class_weights

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
        policy_heads = getattr(self.policy, "policy_heads", None)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if policy_heads is not None:
            params.extend(policy_heads.parameters())
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
        if policy_heads is not None:
            policy_heads.train()
        total_loss, steps = 0.0, 0
        rank_loss_total, evidence_loss_total, action_loss_total, calibration_loss_total = 0.0, 0.0, 0.0, 0.0
        score_pointwise_loss_total = 0.0
        evidence_correct, evidence_total = 0, 0
        action_correct, action_total = 0, 0
        calibration_probs: list[float] = []
        calibration_labels: list[float] = []
        for _ in range(epochs):
            for start in range(0, len(usable), self.batch_size):
                batch = usable[start:start + self.batch_size]
                all_pairs, split, pos_lens = [], [0], []
                evidence_targets = []
                score_weights = []
                action_targets = []
                for ex in batch:
                    pos_ids = list(ex["positive_doc_ids"])
                    neg_ids = list(ex["negative_doc_ids"])
                    pos = [(ex["question"], self._doc_text(ex["documents"], d))
                           for d in pos_ids]
                    neg = [(ex["question"], self._doc_text(ex["documents"], d))
                           for d in neg_ids]
                    pos_weights = ex.get("positive_doc_weights") or {}
                    neg_weights = ex.get("negative_doc_weights") or {}
                    all_pairs.extend(pos + neg)
                    evidence_targets.extend([1.0] * len(pos) + [0.0] * len(neg))
                    score_weights.extend(
                        float(pos_weights.get(str(d), pos_weights.get(d, 1.0)))
                        for d in pos_ids
                    )
                    score_weights.extend(
                        float(neg_weights.get(str(d), neg_weights.get(d, 1.0)))
                        for d in neg_ids
                    )
                    split.append(len(all_pairs))
                    pos_lens.append(len(pos))
                    action_targets.append(ex.get("action_target"))

                inputs = tok(all_pairs, padding=True, truncation=True,
                             return_tensors="pt", max_length=self.max_length).to(device)
                out = model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=policy_heads is not None,
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
                evidence_loss = torch.tensor(0.0, device=device)
                action_loss = torch.tensor(0.0, device=device)
                calibration_loss = torch.tensor(0.0, device=device)
                score_pointwise_loss = torch.tensor(0.0, device=device)

                if self.score_pointwise_loss_weight:
                    target = torch.tensor(evidence_targets, dtype=torch.float32, device=device)
                    weights = torch.tensor(score_weights, dtype=torch.float32, device=device)
                    weights = torch.clamp(weights, min=0.0)
                    raw_pointwise = F.binary_cross_entropy_with_logits(
                        scores.view(-1), target, reduction="none")
                    normalizer = weights.sum().clamp_min(1.0)
                    score_pointwise_loss = (raw_pointwise * weights).sum() / normalizer
                    loss = loss + self.score_pointwise_loss_weight * score_pointwise_loss

                if policy_heads is not None and out.hidden_states:
                    pooled = out.hidden_states[-1][:, 0]
                    head_out = policy_heads(pooled)
                    target = torch.tensor(evidence_targets, dtype=torch.float32, device=device)
                    if self.evidence_loss_weight:
                        evidence_loss = F.binary_cross_entropy_with_logits(
                            head_out["evidence_logits"].view(-1), target)
                        loss = loss + self.evidence_loss_weight * evidence_loss
                    with torch.no_grad():
                        evidence_probs = torch.sigmoid(head_out["evidence_logits"].view(-1))
                        evidence_preds = (evidence_probs >= 0.5).float()
                        evidence_correct += int((evidence_preds == target).sum().item())
                        evidence_total += int(target.numel())
                    if self.calibration_loss_weight:
                        calibration_loss = F.binary_cross_entropy_with_logits(
                            head_out["confidence_logits"].view(-1), target)
                        loss = loss + self.calibration_loss_weight * calibration_loss
                    with torch.no_grad():
                        conf_probs = torch.sigmoid(head_out["confidence_logits"].view(-1))
                        calibration_probs.extend(float(x) for x in conf_probs.detach().cpu().tolist())
                        calibration_labels.extend(float(x) for x in target.detach().cpu().tolist())
                    if self.action_loss_weight:
                        action_rows, action_labels = [], []
                        label_map = getattr(self.policy, "action_label_to_id", {})
                        for i, action in enumerate(action_targets):
                            if action not in label_map:
                                continue
                            s, e = split[i], split[i + 1]
                            action_rows.append(pooled[s:e].mean(dim=0))
                            action_labels.append(label_map[action])
                        if action_rows:
                            action_repr = torch.stack(action_rows, dim=0)
                            action_logits = policy_heads(action_repr)["action_logits"]
                            labels = torch.tensor(action_labels, dtype=torch.long, device=device)
                            class_weights = None
                            if self.action_class_weights is not None:
                                class_weights = torch.tensor(
                                    self.action_class_weights,
                                    dtype=torch.float32,
                                    device=device,
                                )
                            action_loss = F.cross_entropy(
                                action_logits, labels, weight=class_weights)
                            loss = loss + self.action_loss_weight * action_loss
                            with torch.no_grad():
                                preds = torch.argmax(action_logits, dim=-1)
                                action_correct += int((preds == labels).sum().item())
                                action_total += int(labels.numel())

                if (
                    rank_valid == 0
                    and policy_heads is None
                    and not self.score_pointwise_loss_weight
                ):
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                rank_loss_total += float(rank_loss.item())
                evidence_loss_total += float(evidence_loss.item())
                action_loss_total += float(action_loss.item())
                calibration_loss_total += float(calibration_loss.item())
                score_pointwise_loss_total += float(score_pointwise_loss.item())
                steps += 1

        policy_enabled = policy_heads is not None
        if steps and policy_enabled:
            self.policy.policy_heads_loaded = True
        result = {
            "trained_samples": len(usable),
            "skipped": len(pairs) - len(usable),
            "steps": steps,
            "avg_loss": (total_loss / steps) if steps else None,
            "avg_rank_loss": (rank_loss_total / steps) if steps else None,
            "policy_heads_enabled": policy_enabled,
            "avg_evidence_loss": (evidence_loss_total / steps) if (steps and policy_enabled) else None,
            "avg_action_loss": (action_loss_total / steps) if (steps and policy_enabled) else None,
            "avg_calibration_loss": (calibration_loss_total / steps) if (steps and policy_enabled) else None,
            "avg_score_pointwise_loss": (score_pointwise_loss_total / steps) if steps else None,
            "score_pointwise_loss_weight": self.score_pointwise_loss_weight,
            "evidence_accuracy": (evidence_correct / evidence_total) if evidence_total else None,
            "action_accuracy": (action_correct / action_total) if action_total else None,
            "calibration_ece": _binary_ece(calibration_probs, calibration_labels),
        }
        return result

    def save(self, out_dir: str) -> None:
        self.policy.save_lora(out_dir)
