import json

from conftest import make_sample
from evoco_rag.config import EvoCoConfig
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer
from evoco_rag.trainers.large_trainer import LargeTrainer


def test_grpo_config_is_loadable():
    cfg = EvoCoConfig.from_dict({
        "training": {
            "large_train_method": "grpo",
            "grpo_num_generations": 4,
            "grpo_n_per_train": 2,
            "grpo_epochs": 1,
            "grpo_beta": 0.03,
            "grpo_temperature": 0.8,
            "grpo_gradient_accumulation_steps": 2,
            "grpo_max_steps": 5,
        }
    })

    assert cfg.training.large_train_method == "grpo"
    assert cfg.training.grpo_num_generations == 4
    assert cfg.training.grpo_n_per_train == 2
    assert cfg.training.grpo_beta == 0.03
    assert cfg.training.grpo_max_steps == 5


def test_grpo_reward_uses_corag_style_substring_matching():
    reward = LargeTrainer.make_grpo_reward_func()

    scores = reward(
        completions=["The final answer is London.", "I think it is Paris."],
        answers=[["London"], ["Berlin"]],
        sample_id=["a", "b"],
    )

    assert scores == [1.0, 0.0]


def test_coevolution_dispatches_large_grpo(tmp_path, capsys):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.models.small_lora_dir = str(tmp_path / "small")
    cfg.models.large_lora_dir = str(tmp_path / "large")
    cfg.training.large_train_method = "grpo"
    cfg.training.grpo_num_generations = 2
    cfg.training.grpo_n_per_train = 1
    cfg.training.grpo_epochs = 1
    cfg.training.grpo_beta = 0.04
    cfg.training.grpo_temperature = 0.7
    cfg.training.grpo_gradient_accumulation_steps = 1
    cfg.training.grpo_max_steps = 1
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False

    class LargeTrainerStub:
        def __init__(self):
            self.grpo_called = False
            self.sft_called = False
            self.saved = []

        def train_sft(self, experiences, batch_size):
            self.sft_called = True
            return {"method": "sft"}

        def train_grpo(self, experiences, **kwargs):
            self.grpo_called = True
            self.kwargs = kwargs
            return {"method": "grpo", "trained_samples": len(experiences)}

        def save(self, path):
            self.saved.append(path)

    class EvaluatorStub:
        def evaluate(self, round_id):
            return {"num_examples": 1}

        def evaluate_generalization(self, round_id):
            metrics = {"num_examples": 1, "strict_accuracy": 0.0, "corag_style_accuracy": 0.0}
            metrics_dir = tmp_path / "out" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / f"test_eval_round_{round_id:03d}.json").write_text(
                json.dumps(metrics), encoding="utf-8")
            (metrics_dir / f"test_predictions_round_{round_id:03d}.jsonl").write_text(
                "{}\n", encoding="utf-8")
            return metrics

    large_trainer = LargeTrainerStub()
    trainer = CoevolutionTrainer(
        cfg,
        small_policy=None,
        large_auditor=None,
        large_trainer=large_trainer,
        evaluator=EvaluatorStub(),
    )
    stats = trainer.run_round([make_sample()], round_id=0)

    assert large_trainer.grpo_called is True
    assert large_trainer.sft_called is False
    assert stats["large"]["method"] == "grpo"
    assert large_trainer.kwargs["num_generations"] == 2
    assert "large training method: grpo" in capsys.readouterr().out
