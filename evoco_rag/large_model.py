"""大模型 generator + auditor（开发文档 §5.5）。

封装 mistralai/Mistral-Nemo-Instruct-2407：基于证据合约生成答案并审计，输出 LargeAudit。
默认 bf16（与 run_train.py 对齐，H20 显存充足）；use_4bit=True 时走 nf4 量化。
JSON 解析失败最多重试 json_retry 次，仍失败则给 fallback audit、json_valid=False。
torch / transformers / peft 延迟导入。
"""

from __future__ import annotations

from typing import Optional

from .auditor import build_audit_prompt, parse_audit
from .schemas import EvidenceContract, LargeAudit, RagSample


class LargeGeneratorAuditor:
    def __init__(
        self,
        base_path: str,
        lora_dir: Optional[str] = None,
        use_lora: bool = True,
        use_4bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_prompt_length: int = 2048,
        max_completion_length: int = 1024,
        json_retry: int = 3,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.json_retry = json_retry

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
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_completion_length,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def generate_audit(
        self,
        sample: RagSample,
        contract: EvidenceContract,
        show_gold: bool = False,
        round_id: int = 0,
    ) -> tuple[LargeAudit, bool]:
        """生成审计，带 JSON 重试与降级。返回 (LargeAudit, json_valid)。"""
        messages = build_audit_prompt(sample, contract, show_gold=show_gold)
        last_text = ""
        for attempt in range(self.json_retry):
            text = self._generate(messages, temperature=0.0 if attempt == 0 else 0.7)
            last_text = text
            audit, ok = parse_audit(text, sample.sample_id, round_id)
            if ok:
                return audit, True
        # 仍失败：fallback
        audit, ok = parse_audit(last_text, sample.sample_id, round_id)
        return audit, ok

    def save_lora(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
