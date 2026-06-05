"""数据加载与统一样本格式（开发文档 §4.1、§5.2）。

兼容两类现有数据：
  训练：../rag_assets/data_v33/Pop/train_labels_list.json  —— question / answers / context / labels
  测试：../rag_assets/data/Pop/test.json                    —— question / answers / ctxs
统一转为 RagSample。context 字符串形如 "title: X\ncontext: Y"，用简单规则切出
title 与 text，并保留 raw。
"""

from __future__ import annotations

import json
from typing import Optional

from .schemas import RagSample


def _parse_context_string(raw: str) -> tuple[str, str]:
    """从 "title: X\ncontext: Y" 切出 (title, text)；失败则 title 空、text=raw。"""
    title, text = "", raw
    if raw.startswith("title:"):
        rest = raw[len("title:"):]
        if "\ncontext:" in rest:
            t, c = rest.split("\ncontext:", 1)
            title, text = t.strip(), c.strip()
        else:
            title = rest.strip()
            text = ""
    return title, text


def _docs_from_context_list(context: list[str]) -> list[dict]:
    docs = []
    for i, raw in enumerate(context):
        title, text = _parse_context_string(raw)
        docs.append({"doc_id": i, "title": title, "text": text, "raw": raw})
    return docs


def _docs_from_ctxs(ctxs: list[dict]) -> list[dict]:
    docs = []
    for i, ctx in enumerate(ctxs):
        title = (ctx.get("title") or "").strip()
        text = (ctx.get("text") or "").strip()
        raw = f"title: {title}\ncontext: {text}"
        docs.append({"doc_id": i, "title": title, "text": text, "raw": raw})
    return docs


def load_train_samples(
    path: str,
    dataset_name: str = "Pop",
    debug_size: Optional[int] = None,
) -> list[RagSample]:
    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    if debug_size:
        raw_list = raw_list[:debug_size]

    samples = []
    for i, item in enumerate(raw_list):
        docs = _docs_from_context_list(item.get("context", []))
        samples.append(RagSample(
            sample_id=f"{dataset_name.lower()}-train-{i:06d}",
            question=item["question"],
            answers=item.get("answers", []),
            documents=docs,
            seed_labels=item.get("labels"),
            metadata={"dataset": dataset_name, "split": "train"},
        ).validate())
    return samples


def load_test_samples(
    path: str,
    dataset_name: str = "Pop",
    debug_size: Optional[int] = None,
) -> list[RagSample]:
    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    if debug_size:
        raw_list = raw_list[:debug_size]

    samples = []
    for i, item in enumerate(raw_list):
        docs = _docs_from_ctxs(item.get("ctxs", []))
        samples.append(RagSample(
            sample_id=f"{dataset_name.lower()}-test-{i:06d}",
            question=item["question"],
            answers=item.get("answers", []),
            documents=docs,
            metadata={"dataset": dataset_name, "split": "test"},
        ).validate())
    return samples
