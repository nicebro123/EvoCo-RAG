# EvoCo-RAG

Evidence-contract driven small–large model co-evolution for Retrieval-Augmented
Generation.

A small reranker proposes an auditable **evidence contract**; a large model
answers and **audits** whether the evidence truly supports the answer; a rule
verifier and a **decomposed reward** assign responsibility separately to the
retriever and the generator, which are then updated via two LoRA adapters.

This repository is **code only**. Datasets, weights, checkpoints, and outputs
live in a sibling `../rag_assets/` directory and are never committed.

---

## Documentation

Start with the documentation map if you are new to the repo:

| Guide | Use it when you want to… |
|---|---|
| **[docs/README.md](docs/README.md)** | choose the right reproduction / experiment / research document |
| **[docs/DATASETS.md](docs/DATASETS.md)** | download & place the dataset pack and base model weights |
| **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** | install the GPU env and run training / eval / ablations / trends |
| **[docs/TESTING.md](docs/TESTING.md)** | run unit tests and CPU-safe code checks (no GPU) |
| **[configs/experiments/README.md](configs/experiments/README.md)** | inspect official study specs and launcher behavior |

Design references (not needed to reproduce): `docs/协同进化RAG论文构想.md` (paper
idea), `docs/协同进化RAG代码开发文档.md` (engineering design).

---

## Quickstart

```bash
# 1. Clone (code only) and create the asset root
git clone https://github.com/nicebro123/EvoCo-RAG.git
cd EvoCo-RAG && mkdir -p ../rag_assets

# 2. Datasets + weights  -> see docs/DATASETS.md
# 3. Install GPU env      -> see docs/EXPERIMENTS.md

# 4. Smoke test: 16-sample debug run
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_debug.yaml

# 5. Materialize all official studies without starting training
bash scripts/launch_all_experiments.sh --dry-run

# 6. Sanity-check the code without a GPU
python -m pytest -q          # see docs/TESTING.md
```

---

## Method

The original answer-only loop:

```text
reranker selects top1_doc → generator answers → answer hit ⇒ reward top1_doc
```

becomes a responsibility-aware loop:

```text
small model proposes an EvidenceContract
large model answers and audits the evidence
rule verifier checks answer / evidence / citation
decomposed reward assigns responsibility (retriever vs generator)
replay buffer stores structured experience
small LoRA and large LoRA are updated separately
```

Core idea: do **not** reward the retriever when the generator answers correctly
from parametric knowledge while the selected evidence is actually wrong.

---

## Repository Structure

```text
configs/        YAML configs; configs/experiments/ = study specs; configs/local/ = generated (git-ignored)
docs/           the three guides above + design references
evoco_rag/      the EvoCo-RAG package (schemas, contract, verifier, rewards, trainers, evaluation)
scripts/        CLI entrypoints (train/eval/ablations/trends/replay/dataset tooling)
tests/          CPU-safe tests
run_train.py    legacy CoRAG baseline entrypoint
run_test.py     legacy CoRAG baseline evaluator
```

Recommended entrypoints:

```text
scripts/train_evoco.py      co-evolution training
scripts/eval_evoco.py       full test-set evaluation
scripts/run_ablations.py    ablation matrix
scripts/plot_trends.py      multi-round trend summary + plots
scripts/inspect_replay.py   replay buffer inspection
scripts/build_seed_replay.py  CPU-only seed replay
scripts/launch_all_experiments.sh  one-command tmux launcher for official studies
```

The default all-study launcher uses **full-data** configs for every official
study: PopQAStandard sweeps, PopQAStandard hyperparameter exploration,
multi-dataset checks, selected PopQA settings, and PopQA mechanism ablations.
Fast specs remain available for debugging; use `--spec` to run a single study.

---

## References

- BGE reranker: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- Mistral-Nemo-Instruct-2407: <https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407>
- Hugging Face download guide: <https://huggingface.co/docs/huggingface_hub/en/guides/download>
