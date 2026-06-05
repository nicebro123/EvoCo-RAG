"""检查 replay buffer 内容与分布（开发文档 §3 scripts、§12）。

    python scripts/inspect_replay.py --replay ../rag_assets/outputs_debug/latest/replay/round_000.jsonl
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.evaluation.metrics import compute_metrics
from evoco_rag.replay_buffer import ReplayBuffer
from evoco_rag.schemas import ReplayExperience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    args = ap.parse_args()

    exps = []
    with open(args.replay, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                exps.append(ReplayExperience.from_dict(json.loads(line)))
    print(f"total experiences: {len(exps)}")

    ft = collections.Counter(
        e.training_targets.get("failure_type") or e.audit.get("failure_type") for e in exps)
    print("failure_type distribution:")
    for k, v in ft.most_common():
        print(f"  {k}: {v}")

    sft = sum(1 for e in exps if e.training_targets.get("large_sft_eligible"))
    print(f"large_sft_eligible: {sft}")

    print("\ncredit assignment summary:")
    print(json.dumps(ReplayBuffer.credit_assignment_summary(exps), ensure_ascii=False, indent=2))

    print("\naudit trust summary:")
    print(json.dumps(ReplayBuffer.trust_summary(exps), ensure_ascii=False, indent=2))

    print("\naggregate metrics:")
    print(json.dumps(compute_metrics(exps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
