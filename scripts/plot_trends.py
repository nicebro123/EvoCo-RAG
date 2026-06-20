"""多轮协同进化趋势汇总 + 出图。

读取 outputs/metrics/round_*.json（CoevolutionTrainer 每轮写的 stats），抽取每轮
的"真实泛化"评估指标（stats["eval"]，即测试集 show_gold=False）与训练损失，
输出：
  - 控制台对比表
  - <metrics_dir>/trends.json 与 trends.csv
  - <metrics_dir>/trend_metrics.png（accuracy / 证据支持率 / 未支持答案率 / ask_auditor 占比）
  - <metrics_dir>/trend_training.png（大小模型训练损失与 head 准确率）

用法:
    python scripts/plot_trends.py --config configs/evoco_popqa.yaml
    python scripts/plot_trends.py --metrics_dir ../rag_assets/outputs/evoco_popqa/metrics
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROUND_RE = re.compile(r"^round_(\d+)\.json$")

# 从 eval（真实泛化）里抽取的标量指标
EVAL_METRICS = [
    "accuracy", "recall_at_k", "mrr", "evidence_support_rate",
    "citation_correctness", "evidence_quote_support_rate", "used_doc_precision",
    "unsupported_answer_rate",
    "wrong_retriever_reward_rate", "avg_selected_docs", "avg_total_cost_penalty",
    "cost_per_correct_answer", "generator_call_rate", "audit_call_rate",
    "audit_nonempty_output_rate", "avg_generation_candidates", "empty_answer_rate",
    "unfulfilled_action_rate", "audit_schema_valid_rate", "audit_trust_weight_mean",
    "confidence_success_correlation", "ece", "evaluation_protocol_version",
]


def _action_ratio(dist: dict, key: str):
    if not dist:
        return None
    total = sum(dist.values())
    return (dist.get(key, 0) / total) if total else None


def collect_trends(metrics_dir: str) -> list[dict]:
    """读取 metrics_dir 下所有 round_NNN.json，按 round 升序返回每轮汇总行。"""
    rows = []
    for path in glob.glob(os.path.join(metrics_dir, "round_*.json")):
        m = _ROUND_RE.match(os.path.basename(path))
        if not m:
            continue
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        eval_m = stats.get("eval", {}) or {}
        small = stats.get("small", {}) or {}
        large = stats.get("large", {}) or {}
        action_dist = eval_m.get("action_distribution", {}) or {}
        timing = stats.get("timing", {}) or {}

        row = {
            "round": stats.get("round", int(m.group(1))),
            "eval_source": stats.get("eval_source", "unknown"),
            "num_eval_examples": eval_m.get("num_examples"),
            "ask_auditor_ratio": _action_ratio(action_dist, "ask_auditor"),
            "answer_now_ratio": _action_ratio(action_dist, "answer_now"),
            "retrieve_more_ratio": _action_ratio(action_dist, "retrieve_more"),
            "small_avg_loss": small.get("avg_loss"),
            "small_action_accuracy": small.get("action_accuracy"),
            "small_evidence_accuracy": small.get("evidence_accuracy"),
            "small_calibration_ece": small.get("calibration_ece"),
            "large_avg_loss": large.get("avg_loss"),
            "total_round_seconds": timing.get("total_round_seconds"),
        }
        for k in EVAL_METRICS:
            row[k] = eval_m.get(k)
        rows.append(row)
    rows.sort(key=lambda r: r["round"])
    return rows


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(rows: list[dict]) -> None:
    cols = ["round", "eval_source", "accuracy", "evidence_support_rate",
            "unsupported_answer_rate", "citation_correctness", "mrr",
            "ask_auditor_ratio", "avg_total_cost_penalty"]
    header = "".join(f"{c:>24}" if c != "round" else f"{c:>6}" for c in cols)
    print(header)
    for r in rows:
        line = ""
        for c in cols:
            w = 6 if c == "round" else 24
            line += f"{_fmt(r.get(c)):>{w}}"
        print(line)


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _series(rows, key):
    xs = [r["round"] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


def plot_metrics(rows: list[dict], out_path: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("accuracy", "Test accuracy (%)"),
        ("evidence_support_rate", "Evidence support rate"),
        ("unsupported_answer_rate", "Unsupported-answer rate"),
        ("ask_auditor_ratio", "ask_auditor action ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.grid(True, alpha=0.3)
        if xs:
            ax.set_xticks(xs)
    fig.suptitle("EvoCo-RAG co-evolution trends (real generalization)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_training(rows: list[dict], out_path: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [("small_avg_loss", "small avg_loss"),
                       ("large_avg_loss", "large avg_loss")]:
        xs, ys = _series(rows, key)
        if xs:
            ax1.plot(xs, ys, marker="o", label=label)
    ax1.set_title("Training loss")
    ax1.set_xlabel("round")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for key, label in [("small_action_accuracy", "action acc"),
                       ("small_evidence_accuracy", "evidence acc"),
                       ("small_calibration_ece", "head ECE")]:
        xs, ys = _series(rows, key)
        if xs:
            ax2.plot(xs, ys, marker="o", label=label)
    ax2.set_title("Small-model policy heads")
    ax2.set_xlabel("round")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="从 config 推导 metrics 目录")
    ap.add_argument("--metrics_dir", default=None, help="直接指定 metrics 目录")
    ap.add_argument("--no_plot", action="store_true", help="只出表与 CSV，不画图")
    args = ap.parse_args()

    metrics_dir = args.metrics_dir
    if metrics_dir is None:
        if args.config is None:
            raise SystemExit("需要 --metrics_dir 或 --config 之一")
        from evoco_rag.config import EvoCoConfig
        cfg = EvoCoConfig.load(args.config)
        metrics_dir = os.path.join(cfg.output_dir, "metrics")

    if not os.path.isdir(metrics_dir):
        raise SystemExit(f"metrics 目录不存在: {metrics_dir}")

    rows = collect_trends(metrics_dir)
    if not rows:
        raise SystemExit(f"{metrics_dir} 下没有 round_*.json")

    print(f"collected {len(rows)} rounds from {metrics_dir}\n")
    print_table(rows)

    trends_json = os.path.join(metrics_dir, "trends.json")
    trends_csv = os.path.join(metrics_dir, "trends.csv")
    with open(trends_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    write_csv(rows, trends_csv)
    print(f"\nsaved: {trends_json}")
    print(f"saved: {trends_csv}")

    if not args.no_plot:
        try:
            p1 = os.path.join(metrics_dir, "trend_metrics.png")
            p2 = os.path.join(metrics_dir, "trend_training.png")
            plot_metrics(rows, p1)
            plot_training(rows, p2)
            print(f"saved: {p1}")
            print(f"saved: {p2}")
        except ImportError:
            print("matplotlib 不可用，跳过出图（已生成 trends.json/csv）")


if __name__ == "__main__":
    main()
