"""配置加载（开发文档 §8、§13.4）。

所有路径与超参只从 config 读取，核心模块不硬编码数据、权重或输出目录。
支持 yaml（若安装）或 json。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .rewards import RewardWeights


@dataclass
class ContractConfig:
    top_k: int = 5
    high_conf_threshold: float = 0.75
    answer_now_margin: float = 0.15
    max_selected_docs: int = 5
    action_mode: str = "heuristic"  # heuristic | policy | hybrid
    policy_action_min_conf: float = 0.45


@dataclass
class TrainingConfig:
    num_rounds: int = 3
    batch_size: int = 4
    large_batch_size: int = 2
    small_lr: float = 5.0e-5
    large_lr: float = 1.0e-5


@dataclass
class RuntimeConfig:
    candidate_doc_char_limit: int = 1200
    num_audit_candidates: int = 3
    audit_batch_size: int = 1
    audit_temperature: float = 0.7
    max_prompt_length: int = 3072
    max_completion_length: int = 1024
    progress_interval: int = 50
    replay_flush_interval: int = 10


@dataclass
class SmallPolicyConfig:
    use_policy_heads: bool = False
    evidence_loss_weight: float = 1.0
    action_loss_weight: float = 0.5
    calibration_loss_weight: float = 0.2
    answer_now_action_weight: float = 1.0
    retrieve_more_action_weight: float = 1.3
    rewrite_query_action_weight: float = 1.0
    ask_auditor_action_weight: float = 2.0


@dataclass
class ModelsConfig:
    small_base_path: str = "../rag_assets/base_models/reranker/bge-reranker-v2-m3"
    large_base_path: str = "../rag_assets/base_models/generator/Meta-Llama-3-8B-Instruct"
    small_lora_dir: str = "../rag_assets/checkpoints/evoco_popqa/small"
    large_lora_dir: str = "../rag_assets/checkpoints/evoco_popqa/large"
    use_4bit: bool = False


@dataclass
class DataConfig:
    train_path: str = "../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data_v33/Pop/train_labels_list.json"
    test_path: str = "../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data/Pop/test.json"
    dataset_name: str = "PopQAStandard"
    debug_size: int | None = None
    # 每轮训练后做"真实泛化"评估时使用的测试子集大小（None=全量 test）。
    # 全量 test + 大模型审计每轮很贵，故可在训练循环里截断；最终 eval_evoco 仍用全量。
    eval_size: int | None = None


@dataclass
class AblationConfig:
    use_evidence_audit: bool = True
    use_action_policy: bool = True
    use_decomposed_reward: bool = True
    train_small_lora: bool = True
    train_large_lora: bool = True


@dataclass
class EvoCoConfig:
    name: str = "evoco_rag_popqa"
    seed: int = 42
    output_dir: str = "../rag_assets/outputs/evoco_popqa"
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    small_policy: SmallPolicyConfig = field(default_factory=SmallPolicyConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @staticmethod
    def _build_section(section_name: str, section_type, raw: dict):
        if not isinstance(raw, dict):
            raise ValueError(f"config section {section_name!r} must be a mapping")
        allowed = set(section_type.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"unknown config keys in {section_name}: {', '.join(unknown)}")
        return section_type(**raw)

    @classmethod
    def from_dict(cls, d: dict) -> "EvoCoConfig":
        if not isinstance(d, dict):
            raise ValueError("config must contain a mapping")
        allowed_sections = {
            "project", "data", "models", "contract", "training",
            "runtime", "small_policy", "reward", "ablation",
        }
        unknown_sections = sorted(set(d) - allowed_sections)
        if unknown_sections:
            raise ValueError(
                f"unknown top-level config sections: {', '.join(unknown_sections)}")
        proj = d.get("project", {})
        if not isinstance(proj, dict):
            raise ValueError("config section 'project' must be a mapping")
        unknown_project = sorted(set(proj) - {"name", "seed", "output_dir"})
        if unknown_project:
            raise ValueError(
                f"unknown config keys in project: {', '.join(unknown_project)}")
        reward_raw = d.get("reward", {})
        return cls(
            name=proj.get("name", "evoco_rag_popqa"),
            seed=proj.get("seed", 42),
            output_dir=proj.get("output_dir", "../rag_assets/outputs/evoco_popqa"),
            data=cls._build_section("data", DataConfig, d.get("data", {})),
            models=cls._build_section("models", ModelsConfig, d.get("models", {})),
            contract=cls._build_section(
                "contract", ContractConfig, d.get("contract", {})),
            training=cls._build_section(
                "training", TrainingConfig, d.get("training", {})),
            runtime=cls._build_section("runtime", RuntimeConfig, d.get("runtime", {})),
            small_policy=cls._build_section(
                "small_policy", SmallPolicyConfig, d.get("small_policy", {})),
            reward=cls._build_section("reward", RewardWeights, reward_raw),
            ablation=cls._build_section(
                "ablation", AblationConfig, d.get("ablation", {})),
        )

    @classmethod
    def load(cls, path: str) -> "EvoCoConfig":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            import yaml  # 延迟导入
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw or {})
