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
    num_generations: int = 2
    small_lr: float = 5.0e-5
    large_lr: float = 1.0e-5
    train_small_lora: bool = True
    train_large_lora: bool = True


@dataclass
class RuntimeConfig:
    candidate_doc_char_limit: int = 1200
    num_audit_candidates: int = 3
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


@dataclass
class ModelsConfig:
    small_base_path: str = "../rag_assets/base_models/reranker/bge-reranker-v2-m3"
    large_base_path: str = "../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407"
    small_lora_dir: str = "../rag_assets/checkpoints/evoco_popqa/small"
    large_lora_dir: str = "../rag_assets/checkpoints/evoco_popqa/large"
    use_4bit: bool = False


@dataclass
class DataConfig:
    train_path: str = "../rag_assets/data_v33/Pop/train_labels_list.json"
    test_path: str = "../rag_assets/data/Pop/test.json"
    dataset_name: str = "Pop"
    debug_size: int | None = None


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

    @classmethod
    def from_dict(cls, d: dict) -> "EvoCoConfig":
        proj = d.get("project", {})
        reward_raw = d.get("reward", {})
        return cls(
            name=proj.get("name", "evoco_rag_popqa"),
            seed=proj.get("seed", 42),
            output_dir=proj.get("output_dir", "../rag_assets/outputs/evoco_popqa"),
            data=DataConfig(**{k: v for k, v in d.get("data", {}).items()
                               if k in DataConfig.__dataclass_fields__}),
            models=ModelsConfig(**{k: v for k, v in d.get("models", {}).items()
                                   if k in ModelsConfig.__dataclass_fields__}),
            contract=ContractConfig(**{k: v for k, v in d.get("contract", {}).items()
                                       if k in ContractConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in d.get("training", {}).items()
                                       if k in TrainingConfig.__dataclass_fields__}),
            runtime=RuntimeConfig(**{k: v for k, v in d.get("runtime", {}).items()
                                     if k in RuntimeConfig.__dataclass_fields__}),
            small_policy=SmallPolicyConfig(**{k: v for k, v in d.get("small_policy", {}).items()
                                              if k in SmallPolicyConfig.__dataclass_fields__}),
            reward=RewardWeights(**{k: v for k, v in reward_raw.items()
                                    if k in RewardWeights.__dataclass_fields__}),
            ablation=AblationConfig(**{k: v for k, v in d.get("ablation", {}).items()
                                       if k in AblationConfig.__dataclass_fields__}),
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
