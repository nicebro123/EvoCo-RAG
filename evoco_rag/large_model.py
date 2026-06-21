"""大模型 generator + auditor（开发文档 §5.5）。

封装 mistralai/Mistral-Nemo-Instruct-2407：基于证据合约生成答案并审计，输出 LargeAudit。
默认 bf16（与 run_train.py 对齐，H20 显存充足）；use_4bit=True 时走 nf4 量化。
JSON 解析失败最多重试 json_retry 次，仍失败则给 fallback audit、json_valid=False。
torch / transformers / peft 延迟导入。
"""

from __future__ import annotations

from collections import Counter
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
        audit_batch_size: int = 1,
        audit_temperature: float = 0.7,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.json_retry = json_retry
        self.candidate_doc_char_limit = max(1, int(candidate_doc_char_limit))
        self.num_audit_candidates = max(1, int(num_audit_candidates))
        self.audit_batch_size = max(1, int(audit_batch_size))
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

        self.model.eval()
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
                # Some chat models ship sampling defaults in generation_config.
                # Clear them explicitly for greedy decoding to avoid noisy warnings.
                gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def _generate_many(self, messages_batch: list[list[dict]], temperature: float) -> list[str]:
        if not messages_batch:
            return []
        import torch

        old_padding_side = getattr(self.tokenizer, "padding_side", "left")
        self.tokenizer.padding_side = "left"
        try:
            self.model.eval()
            prompts = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_batch
            ]
            inputs = self.tokenizer(
                prompts, padding=True, return_tensors="pt", truncation=True,
                max_length=self.max_prompt_length).to(self.model.device)
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=self.max_completion_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
                else:
                    # Some chat models ship sampling defaults in generation_config.
                    # Clear them explicitly for greedy decoding to avoid noisy warnings.
                    gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
                out = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            prompt_width = inputs["input_ids"].shape[1]
            generated = [row[prompt_width:] for row in out]
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        finally:
            self.tokenizer.padding_side = old_padding_side

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

    def _build_audit_messages(
        self,
        sample: RagSample,
        contract: EvidenceContract,
        show_gold: bool = False,
    ) -> list[dict]:
        return build_audit_prompt(
            sample,
            contract,
            show_gold=show_gold,
            candidate_doc_char_limit=self.candidate_doc_char_limit,
        )

    def _candidate_from_text(
        self,
        sample: RagSample,
        contract: EvidenceContract,
        text: str,
        round_id: int,
        attempt: int,
        temperature: float,
    ) -> dict:
        audit, ok = parse_audit(text, sample.sample_id, round_id)
        score = self.score_audit_candidate(sample, contract, audit, ok)
        return {
            "audit": audit,
            "json_valid": ok,
            "score": score,
            "attempt": attempt,
            "temperature": temperature,
            "raw_text": text,
        }

    def _select_best_candidate(
        self,
        sample: RagSample,
        contract: EvidenceContract,
        candidates: list[dict],
    ) -> tuple[LargeAudit, bool]:
        best = max(candidates, key=lambda x: (x["score"], x["json_valid"], -x["attempt"]))

        valid_candidates = [c for c in candidates if c["json_valid"]]
        answers = [
            (c["audit"].final_answer or "").strip().lower()
            for c in valid_candidates
            if c["audit"].final_answer
        ]
        support_levels = [c["audit"].support_level for c in valid_candidates]
        failure_types = [c["audit"].failure_type for c in valid_candidates]

        def majority_ratio(values: list[str]) -> float:
            if not values:
                return 0.0
            return Counter(values).most_common(1)[0][1] / len(values)

        self_consistency = round(
            (
                majority_ratio(answers)
                + majority_ratio(support_levels)
                + majority_ratio(failure_types)
            ) / 3.0,
            4,
        )
        summaries = []
        for c in candidates:
            a = c["audit"]
            summaries.append({
                "attempt": c["attempt"],
                "temperature": c["temperature"],
                "json_valid": c["json_valid"],
                "score": c["score"],
                "final_answer": a.final_answer,
                "used_doc_ids": a.used_doc_ids,
                "support_level": a.support_level,
                "failure_type": a.failure_type,
                "raw_text": (c["raw_text"] or "")[:2000],
            })

        best_audit = best["audit"]
        best_audit.audit_metadata = {
            **(best_audit.audit_metadata or {}),
            "num_candidates": len(candidates),
            "generator_called": True,
            "generation_candidate_count": len(candidates),
            "extra_audit_called": len(candidates) > 1,
            "requested_action": contract.retrieval_action,
            "action_executed": (
                contract.retrieval_action in {"answer_now", "ask_auditor"}
                or bool(contract.uncertainty.get("retrieval_expanded"))
            ),
            "action_fallback": (
                contract.retrieval_action == "rewrite_query"
                or (
                    contract.retrieval_action == "retrieve_more"
                    and not contract.uncertainty.get("retrieval_expanded")
                )
            ),
            "selected_attempt": best["attempt"],
            "selected_candidate_score": best["score"],
            "self_consistency": self_consistency,
            "candidate_summaries": summaries,
        }
        return best_audit, best["json_valid"]

    def generate_audit_batch(
        self,
        samples: list[RagSample],
        contracts: list[EvidenceContract],
        show_gold: bool = False,
        round_id: int = 0,
        batch_size: int | None = None,
        candidate_counts: list[int] | None = None,
    ) -> list[tuple[LargeAudit, bool]]:
        """Batch audit generation. Candidate attempts stay semantically identical to generate_audit."""
        if len(samples) != len(contracts):
            raise ValueError("samples and contracts must have the same length")
        if not samples:
            return []

        messages = [
            self._build_audit_messages(sample, contract, show_gold=show_gold)
            for sample, contract in zip(samples, contracts)
        ]
        candidates: list[list[dict]] = [[] for _ in samples]
        effective_batch = max(1, int(batch_size or self.audit_batch_size))
        if candidate_counts is None:
            planned_counts = [self.num_audit_candidates] * len(samples)
        else:
            if len(candidate_counts) != len(samples):
                raise ValueError("candidate_counts must match samples")
            planned_counts = [max(1, int(value)) for value in candidate_counts]

        total_candidates = max(planned_counts)
        for attempt in range(total_candidates):
            temperature = 0.0 if attempt == 0 else self.audit_temperature
            active = [idx for idx, count in enumerate(planned_counts) if attempt < count]
            for start in range(0, len(active), effective_batch):
                chunk = active[start:start + effective_batch]
                texts = self._generate_many(
                    [messages[idx] for idx in chunk],
                    temperature=temperature,
                )
                if len(texts) != len(chunk):
                    raise RuntimeError(
                        "batch audit generation returned a different number of outputs")
                for offset, text in enumerate(texts):
                    idx = chunk[offset]
                    candidates[idx].append(self._candidate_from_text(
                        samples[idx], contracts[idx], text, round_id, attempt, temperature))

        attempts_done = list(planned_counts)
        retry_indices = [
            idx for idx, sample_candidates in enumerate(candidates)
            if not any(c["json_valid"] for c in sample_candidates)
            and attempts_done[idx] < self.json_retry
        ]
        while retry_indices:
            next_retry_indices = []
            for start in range(0, len(retry_indices), effective_batch):
                chunk = retry_indices[start:start + effective_batch]
                texts = self._generate_many(
                    [messages[idx] for idx in chunk],
                    temperature=self.audit_temperature,
                )
                if len(texts) != len(chunk):
                    raise RuntimeError(
                        "batch audit retry returned a different number of outputs")
                for offset, text in enumerate(texts):
                    idx = chunk[offset]
                    attempt = attempts_done[idx]
                    candidates[idx].append(self._candidate_from_text(
                        samples[idx], contracts[idx], text, round_id, attempt,
                        self.audit_temperature))
                    attempts_done[idx] += 1
                    if (
                        not any(c["json_valid"] for c in candidates[idx])
                        and attempts_done[idx] < self.json_retry
                    ):
                        next_retry_indices.append(idx)
            retry_indices = next_retry_indices

        return [
            self._select_best_candidate(sample, contract, sample_candidates)
            for sample, contract, sample_candidates in zip(samples, contracts, candidates)
        ]

    def generate_audit(
        self,
        sample: RagSample,
        contract: EvidenceContract,
        show_gold: bool = False,
        round_id: int = 0,
        candidate_count: int | None = None,
    ) -> tuple[LargeAudit, bool]:
        """生成审计，带 JSON 重试与降级。返回 (LargeAudit, json_valid)。"""
        return self.generate_audit_batch(
            [sample],
            [contract],
            show_gold=show_gold,
            round_id=round_id,
            batch_size=1,
            candidate_counts=([candidate_count] if candidate_count is not None else None),
        )[0]

    def save_lora(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
