"""小模型 RAG policy（开发文档 §5.3）。

封装 bge-reranker-v2-m3：对 (question, doc) 打分、产出 top-k，再交给
contract.build_contract 封装成 EvidenceContract。torch / transformers / peft
延迟导入，保证本模块在无 GPU 环境也能被 import。

第一阶段不引入 evidence/action head；置信度由分数经 sigmoid 标定，action 由
启发式规则决定（见 contract.py）。predict_action / predict_evidence_confidence
提供稳定接口，后续可替换为可训练 head。
"""

from __future__ import annotations

from typing import Optional

from .contract import build_contract
from .schemas import EvidenceContract, RagSample
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
        """对样本所有文档打分，返回按分数降序的 [{doc_id, score}]。"""
        import torch

        pairs = [(sample.question, doc.get("text") or doc.get("raw") or "")
                 for doc in sample.documents]
        if not pairs:
            return []
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length,
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
        ranked = [{"doc_id": doc["doc_id"], "score": float(s)}
                  for doc, s in zip(sample.documents, scores)]
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
    ) -> EvidenceContract:
        ranked = self.rank_documents(sample)
        return build_contract(
            sample, ranked, round_id=round_id, top_k=top_k,
            high_conf_threshold=high_conf_threshold,
            answer_now_margin=answer_now_margin,
            max_selected_docs=max_selected_docs,
        )

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
