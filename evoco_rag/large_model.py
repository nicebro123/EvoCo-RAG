"""大模型 generator + auditor（开发文档 §5.5）。

封装 mistralai/Mistral-Nemo-Instruct-2407：基于证据合约生成答案并审计，输出 LargeAudit。
默认 bf16（与 run_train.py 对齐，H20 显存充足）；use_4bit=True 时走 nf4 量化。
JSON 解析失败最多重试 json_retry 次，仍失败则给 fallback audit、json_valid=False。
torch / transformers / peft 延迟导入。
"""

from __future__ import annotations

from typing import Optional

from .auditor import build_audit_prompt, parse_audit
from .schemas import EvidenceContract, FailureType, LargeAudit, RagSample, SupportLevel


class LargeGeneratorAuditor:
    def __init__(
        self,
        base_path: str,
        lora_dir: Optional[str] = None,
        use_lora: bool = True,
        use_4bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_prompt_length: int = 3072,
        max_completion_length: int = 1024,
        json_retry: int = 3,
        candidate_doc_char_limit: int = 1200,
        num_audit_candidates: int = 3,
        audit_temperature: float = 0.7,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.json_retry = json_retry
        self.candidate_doc_char_limit = max(1, int(candidate_doc_char_limit))
        self.num_audit_candidates = max(1, int(num_audit_candidates))
        self.audit_temperature = max(0.0, float(audit_temperature))

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            model = AutoModelForCausalLM.from_pretrained(
                base_path, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

        if lora_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir, is_trainable=use_lora)
        elif use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if use_4bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg)
        self.model = model

    def _generate(self, messages: list[dict], temperature: float) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_prompt_length).to(self.model.device)
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=self.max_completion_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                gen_kwargs["do_sample"] = False
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    @staticmethod
    def _quote_supported(sample: RagSample, evidence: dict) -> bool:
        try:
            doc_id = int(evidence.get("doc_id"))
        except (TypeError, ValueError):
            return False
        quote = str(evidence.get("quote") or "").strip()
        if len(quote) < 8:
            return False
        doc = sample.doc_by_id(doc_id) or {}
        text = (doc.get("text") or doc.get("raw") or "").lower()
        return quote.lower() in text

    @staticmethod
    def score_audit_candidate(
        sample: RagSample,
        contract: EvidenceContract,
        audit: LargeAudit,
        json_valid: bool,
    ) -> float:
        """Score an audit without gold answers for inference-time answer selection."""
        contract_ids = set(contract.selected_doc_ids()) | set(contract.candidate_doc_ids())
        selected_ids = set(contract.selected_doc_ids())
        used_ids = [doc_id for doc_id in audit.used_doc_ids if isinstance(doc_id, int)]

        score = 0.0
        score += 2.0 if json_valid else -2.0

        answer = (audit.final_answer or "").strip()
        if answer:
            score += 1.0
            answer_len = len(answer.split())
            if answer_len <= 12:
                score += 0.5
            elif answer_len > 32:
                score -= 0.5
        else:
            score -= 1.0

        support_scores = {
            SupportLevel.FULLY: 2.0,
            SupportLevel.PARTIALLY: 0.75,
            SupportLevel.UNSUPPORTED: -1.0,
        }
        score += support_scores.get(audit.support_level, -1.0)

        if audit.failure_type == FailureType.NONE:
            score += 1.0
        elif audit.failure_type in {FailureType.UNSUPPORTED_ANSWER, FailureType.GENERATION_ERROR}:
            score -= 0.75
        else:
            score -= 0.25

        if used_ids:
            score += 0.5
            if all(doc_id in contract_ids for doc_id in used_ids):
                score += 1.0
            else:
                score -= 1.0
            if any(doc_id in selected_ids for doc_id in used_ids):
                score += 0.5
        else:
            score -= 1.0

        evidence = audit.used_evidence or []
        if evidence:
            hits = sum(1 for item in evidence if LargeGeneratorAuditor._quote_supported(sample, item))
            score += min(1.0, hits / max(1, len(evidence)))
            if hits == 0:
                score -= 0.5

        return round(score, 4)

    def generate_audit(
        self,
        sample: RagSample,
        contract: EvidenceContract,
        show_gold: bool = False,
        round_id: int = 0,
    ) -> tuple[LargeAudit, bool]:
        """生成审计，带 JSON 重试与降级。返回 (LargeAudit, json_valid)。"""
        messages = build_audit_prompt(
            sample,
            contract,
            show_gold=show_gold,
            candidate_doc_char_limit=self.candidate_doc_char_limit,
        )
        candidates: list[tuple[LargeAudit, bool, float, int]] = []
        total_candidates = self.num_audit_candidates
        for attempt in range(total_candidates):
            temperature = 0.0 if attempt == 0 else self.audit_temperature
            text = self._generate(messages, temperature=temperature)
            audit, ok = parse_audit(text, sample.sample_id, round_id)
            score = self.score_audit_candidate(sample, contract, audit, ok)
            candidates.append((audit, ok, score, attempt))

        extra_retries = max(0, self.json_retry - total_candidates)
        for retry in range(extra_retries):
            if any(ok for _, ok, _, _ in candidates):
                break
            text = self._generate(messages, temperature=self.audit_temperature)
            audit, ok = parse_audit(text, sample.sample_id, round_id)
            score = self.score_audit_candidate(sample, contract, audit, ok)
            candidates.append((audit, ok, score, total_candidates + retry))

        best = max(candidates, key=lambda x: (x[2], x[1], -x[3]))
        return best[0], best[1]

    def save_lora(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
