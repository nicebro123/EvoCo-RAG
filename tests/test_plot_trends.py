"""趋势汇总脚本的纯逻辑测试。"""

import importlib.util
import json
import os

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts", "plot_trends.py")
_spec = importlib.util.spec_from_file_location("plot_trends", _SCRIPT)
plot_trends = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_trends)


def _round_stats(round_id, accuracy, retrieve_more, support_rate):
    return {
        "round": round_id,
        "eval_source": "test_generalization",
        "eval": {
            "num_examples": 100,
            "accuracy": accuracy,
            "evidence_support_rate": support_rate,
            "unsupported_answer_rate": 0.05,
            "citation_correctness": 0.7,
            "mrr": 0.8,
            "avg_total_cost_penalty": 0.3,
            "action_distribution": {"answer_now": 100 - retrieve_more,
                                    "retrieve_more": retrieve_more},
        },
        "small": {"avg_loss": 1.5},
        "large": {"avg_loss": 0.12},
        "timing": {"total_round_seconds": 100.0},
    }


def _write_round(metrics_dir, stats):
    path = os.path.join(metrics_dir, f"round_{stats['round']:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def test_action_ratio():
    assert plot_trends._action_ratio({"answer_now": 3, "retrieve_more": 1}, "retrieve_more") == 0.25
    assert plot_trends._action_ratio({}, "retrieve_more") is None
    assert plot_trends._action_ratio({"answer_now": 0}, "retrieve_more") is None


def test_collect_trends_sorted_and_extracted(tmp_path):
    md = str(tmp_path)
    # 故意乱序写入
    _write_round(md, _round_stats(1, 70.0, 20, 0.8))
    _write_round(md, _round_stats(0, 60.0, 40, 0.7))
    # 干扰文件：不应被收集
    _write_round_noise = os.path.join(md, "test_eval_round_000.json")
    with open(_write_round_noise, "w", encoding="utf-8") as f:
        json.dump({"accuracy": 999}, f)

    rows = plot_trends.collect_trends(md)
    assert [r["round"] for r in rows] == [0, 1]
    assert rows[0]["accuracy"] == 60.0
    assert rows[1]["accuracy"] == 70.0
    assert rows[0]["retrieve_more_ratio"] == 0.4
    assert rows[1]["retrieve_more_ratio"] == 0.2
    assert rows[0]["eval_source"] == "test_generalization"


def test_plots_written(tmp_path):
    md = str(tmp_path)
    _write_round(md, _round_stats(0, 60.0, 40, 0.7))
    _write_round(md, _round_stats(1, 72.0, 18, 0.82))
    rows = plot_trends.collect_trends(md)

    p1 = os.path.join(md, "trend_metrics.png")
    p2 = os.path.join(md, "trend_training.png")
    plot_trends.plot_metrics(rows, p1)
    plot_trends.plot_training(rows, p2)
    assert os.path.getsize(p1) > 0
    assert os.path.getsize(p2) > 0

    csv_path = os.path.join(md, "trends.csv")
    plot_trends.write_csv(rows, csv_path)
    assert os.path.exists(csv_path)
