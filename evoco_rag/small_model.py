"""小模型 RAG policy（开发文档 §5.3）。

封装 bge-reranker-v2-m3：对 (question, doc) 打分、产出 top-k，再交给
contract.build_contract 封装成 EvidenceContract。torch / transformers / peft
延迟导入，保证本模块在无 GPU 环境也能被 import。

默认路径仍不启用 evidence/action head；置信度由分数经 sigmoid 标定，action 由
启发式规则决定（见 contract.py）。ECR-1 可通过 use_policy_heads=True 启用
evidence/action/confidence heads，训练逻辑在 small_trainer.py。
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

from .contract import build_contract
from .schemas import EvidenceContract, RagSample, RetrievalAction
from .text_utils import best_evidence_span


ACTION_LABELS = [
    RetrievalAction.ANSWER_NOW,
    RetrievalAction.RETRIEVE_MORE,
    RetrievalAction.REWRITE_QUERY,
    RetrievalAction.ASK_AUDITOR,
]
POLICY_HEADS_FILE = "small_policy_heads.pt"
POLICY_HEADS_CONFIG = "small_policy_heads_config.json"


class SmallPolicyHeads:
    """Lazy torch factory for ECR-1 evidence/action/confidence heads."""

    def __new__(cls, hidden_size: int, num_actions: int):
        import torch.nn as nn

        class _Heads(nn.Module):
            def __init__(self):
                super().__init__()
                self.evidence_head = nn.Linear(hidden_size, 1)
                self.action_head = nn.Linear(hidden_size, num_actions)
                self.confidence_head = nn.Linear(hidden_size, 1)

            def forward(self, pooled):
                return {
                    "evidence_logits": self.evidence_head(pooled).squeeze(-1),
                    "action_logits": self.action_head(pooled),
                    "confidence_logits": self.confidence_head(pooled).squeeze(-1),
                }

        return _Heads()


class SmallRagPolicy:
    def __init__(
        self,
        base_path: str,
        lora_dir: Optional[str] = None,
        use_lora: bool = True,
        lora_r: int = 4,
        lora_alpha: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        use_policy_heads: bool = False,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_policy_heads = use_policy_heads
        self.action_labels = ACTION_LABELS
        self.action_label_to_id = {label: i for i, label in enumerate(self.action_labels)}
        self.last_policy_prediction: dict = {}
        self.policy_heads_loaded = False
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForSequenceClassification.from_pretrained(base_path).to(self.device)

        if lora_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir, is_trainable=use_lora)
        elif use_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "attention.output.dense",
                                "intermediate.dense", "output.dense"],
                lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
            )
            model = get_peft_model(model, cfg)
            for name, param in model.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
        self.model = model
        self.policy_heads = None
        if use_policy_heads:
            hidden_size = self._infer_hidden_size(model)
            if hidden_size is None:
                raise ValueError("model.config.hidden_size is required for SmallPolicyHeads")
            self.policy_heads = SmallPolicyHeads(hidden_size, len(self.action_labels)).to(self.device)
            heads_path = os.path.join(lora_dir or "", POLICY_HEADS_FILE) if lora_dir else ""
            if heads_path and os.path.exists(heads_path):
                self.policy_heads.load_state_dict(torch.load(heads_path, map_location=self.device))
                self.policy_heads_loaded = True
            elif lora_dir:
                warnings.warn(
                    f"use_policy_heads=True but {POLICY_HEADS_FILE} was not found in {lora_dir}; "
                    "initializing fresh policy heads.",
                    RuntimeWarning,
                )

    @staticmethod
    def _infer_hidden_size(model) -> Optional[int]:
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
        base = getattr(model, "base_model", None)
        base_config = getattr(base, "config", None)
        return getattr(base_config, "hidden_size", None)

    def rank_documents(self, sample: RagSample, top_k: Optional[int] = None) -> list[dict]:
        """对样本所有文档打分，返回按分数降序的 [{doc_id, score}]。"""
        import torch

        pairs = [(sample.question, doc.get("text") or doc.get("raw") or "")
                 for doc in sample.documents]
        if not pairs:
            return []
        self.model.eval()
        if self.policy_heads is not None:
            self.policy_heads.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length,
            ).to(self.device)
            out = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=self.policy_heads is not None,
            )
            scores = out.logits.view(-1).float().cpu().tolist()
            evidence_confidences = None
            confidence_scores = None
            if self.policy_heads is not None and out.hidden_states:
                pooled = out.hidden_states[-1][:, 0]
                head_out = self.policy_heads(pooled)
                evidence_confidences = torch.sigmoid(
                    head_out["evidence_logits"]).float().cpu().tolist()
                confidence_scores = torch.sigmoid(
                    head_out["confidence_logits"]).float().cpu().tolist()
                action_logits = head_out["action_logits"].mean(dim=0).float().cpu()
                action_probs = torch.softmax(action_logits, dim=-1)
                action_id = int(torch.argmax(action_logits).item())
                self.last_policy_prediction = {
                    "action": self.action_labels[action_id],
                    "action_logits": [round(float(x), 4) for x in action_logits.tolist()],
                    "action_probs": [round(float(x), 4) for x in action_probs.tolist()],
                    "action_confidence": round(float(action_probs[action_id].item()), 4),
                }
            else:
                self.last_policy_prediction = {}
        ranked = []
        for i, (doc, score) in enumerate(zip(sample.documents, scores)):
            item = {"doc_id": doc["doc_id"], "score": float(score)}
            if evidence_confidences is not None:
                item["evidence_confidence"] = float(evidence_confidences[i])
            if confidence_scores is not None:
                item["policy_confidence"] = float(confidence_scores[i])
            ranked.append(item)
        ranked.sort(key=lambda d: d["score"], reverse=True)
        return ranked[:top_k] if top_k else ranked

    def build_contract(
        self,
        sample: RagSample,
        round_id: int = 0,
        top_k: int = 3,
        high_conf_threshold: float = 0.75,
        answer_now_margin: float = 0.15,
        max_selected_docs: int = 3,
        action_mode: str = "heuristic",
        policy_action_min_conf: float = 0.45,
    ) -> EvidenceContract:
        ranked = self.rank_documents(sample)
        policy_action = self.last_policy_prediction.get("action")
        policy_action_confidence = self.last_policy_prediction.get("action_confidence")
        contract = build_contract(
            sample, ranked, round_id=round_id, top_k=top_k,
            high_conf_threshold=high_conf_threshold,
            answer_now_margin=answer_now_margin,
            max_selected_docs=max_selected_docs,
            action_mode=action_mode,
            policy_action=policy_action,
            policy_action_confidence=policy_action_confidence,
            policy_action_min_conf=policy_action_min_conf,
        )
        if self.last_policy_prediction:
            contract.uncertainty["policy_predicted_action"] = self.last_policy_prediction["action"]
            contract.uncertainty["policy_action_logits"] = self.last_policy_prediction["action_logits"]
            contract.uncertainty["policy_action_probs"] = self.last_policy_prediction["action_probs"]
            contract.uncertainty["policy_heads_loaded"] = self.policy_heads_loaded
        return contract

    def predict_evidence_confidence(self, sample: RagSample, doc: dict) -> float:
        """Heuristic document/span evidence confidence for the first implementation.

        A later trainable evidence head can replace this method without changing
        contract/reward/replay callers.
        """
        text = doc.get("text") or doc.get("raw") or ""
        _, overlap = best_evidence_span(sample.question, text)
        return float(overlap)

    def predict_action(self, contract: EvidenceContract) -> str:
        """Return the current contract action.

        The first implementation uses contract.py heuristics; this method is the
        stable interface for a future trainable action head.
        """
        return contract.retrieval_action

    def save_lora(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
        if self.policy_heads is not None:
            import torch
            os.makedirs(out_dir, exist_ok=True)
            torch.save(self.policy_heads.state_dict(), os.path.join(out_dir, POLICY_HEADS_FILE))
            meta = {
                "action_labels": self.action_labels,
                "use_policy_heads": True,
                "policy_heads_file": POLICY_HEADS_FILE,
            }
            with open(os.path.join(out_dir, POLICY_HEADS_CONFIG), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
