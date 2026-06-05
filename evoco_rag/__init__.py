"""EvoCo-RAG: 证据合约驱动的大小模型协同进化 RAG.

包结构按《EvoCo-RAG 代码开发文档》§3 组织。

核心层（schemas / data / contract / verifier / rewards / replay_buffer）
不依赖 torch，可在纯 CPU 环境 import 与单元测试；
模型相关层（small_model / large_model / auditor / trainers）对 torch
做延迟导入，只有在真正调用时才加载重依赖。
"""

from .schemas import (
    Answerability,
    AnswerCorrectness,
    EvidenceContract,
    EvidenceItem,
    FailureType,
    FeedbackLabel,
    LargeAudit,
    RagSample,
    ReplayExperience,
    RetrievalAction,
    RewardBreakdown,
    RuleVerification,
    SupportLevel,
)
from .weights import (
    adapter_rounds,
    checkpoint_round_dir,
    is_lora_adapter_dir,
    latest_checkpoint_round,
    latest_round_adapter,
    prepare_weight_layout,
    resolve_adapter_for_loading,
    write_weight_manifest,
)

__all__ = [
    "Answerability",
    "AnswerCorrectness",
    "EvidenceContract",
    "EvidenceItem",
    "FailureType",
    "FeedbackLabel",
    "LargeAudit",
    "RagSample",
    "ReplayExperience",
    "RetrievalAction",
    "RewardBreakdown",
    "RuleVerification",
    "SupportLevel",
    "adapter_rounds",
    "checkpoint_round_dir",
    "is_lora_adapter_dir",
    "latest_checkpoint_round",
    "latest_round_adapter",
    "prepare_weight_layout",
    "resolve_adapter_for_loading",
    "write_weight_manifest",
]
