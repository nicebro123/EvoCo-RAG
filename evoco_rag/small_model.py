"""Small reranker used by the RAG pipeline.

The small model's mainline role is intentionally narrow: score
``(question, document)`` pairs, rank candidate documents, and package the
top evidence into an :class:`EvidenceContract`.  It is not an agent and does
not learn "ask/verify" actions; the large model is always responsible for
generation and evidence auditing.
"""

from __future__ import annotations

from typing import Optional

from .contract import build_contract
from .schemas import EvidenceContract, RagSample, RetrievalAction
from .text_utils import best_evidence_span


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
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_policy_prediction: dict = {}
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

    def rank_documents(self, sample: RagSample, top_k: Optional[int] = None) -> list[dict]:
        """Score all candidate documents and return them in descending order."""
        import torch

        pairs = [(sample.question, doc.get("text") or doc.get("raw") or "")
                 for doc in sample.documents]
        if not pairs:
            return []
        self.model.eval()
        self.last_policy_prediction = {}
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length,
            ).to(self.device)
            out = self.model(**inputs)
            scores = out.logits.view(-1).float().cpu().tolist()
        ranked = [
            {"doc_id": doc["doc_id"], "score": float(score)}
            for doc, score in zip(sample.documents, scores)
        ]
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
        retrieve_more_conf_threshold: float | None = None,
        retrieve_more_margin_threshold: float | None = None,
    ) -> EvidenceContract:
        """Build an evidence contract from reranker scores.

        ``retrieve_more`` is retained only as an internal evidence-budget
        expansion signal.  It never means "ask the large model"; the large
        model auditor is invoked by the pipeline regardless of this field.
        """
        ranked = self.rank_documents(sample)
        contract = build_contract(
            sample, ranked, round_id=round_id, top_k=top_k,
            high_conf_threshold=high_conf_threshold,
            answer_now_margin=answer_now_margin,
            max_selected_docs=max_selected_docs,
            retrieve_more_conf_threshold=retrieve_more_conf_threshold,
            retrieve_more_margin_threshold=retrieve_more_margin_threshold,
        )
        if (
            contract.retrieval_action == RetrievalAction.RETRIEVE_MORE
            and len(ranked) > top_k
        ):
            expanded_top_k = min(len(ranked), max(top_k + 1, top_k * 2))
            contract = build_contract(
                sample,
                ranked,
                round_id=round_id,
                top_k=expanded_top_k,
                high_conf_threshold=high_conf_threshold,
                answer_now_margin=answer_now_margin,
                max_selected_docs=max(max_selected_docs, expanded_top_k),
                num_retrieval_rounds=2,
                retrieve_more_conf_threshold=retrieve_more_conf_threshold,
                retrieve_more_margin_threshold=retrieve_more_margin_threshold,
            )
            contract.retrieval_action = RetrievalAction.RETRIEVE_MORE
            contract.uncertainty.update({
                "retrieval_expanded": True,
                "initial_top_k": top_k,
                "effective_top_k": expanded_top_k,
            })
        return contract

    def predict_evidence_confidence(self, sample: RagSample, doc: dict) -> float:
        text = doc.get("text") or doc.get("raw") or ""
        _, overlap = best_evidence_span(sample.question, text)
        return float(overlap)

    def predict_action(self, contract: EvidenceContract) -> str:
        """Legacy compatibility helper; returns the evidence-budget status."""
        return contract.retrieval_action

    def save_lora(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
