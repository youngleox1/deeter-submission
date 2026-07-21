from src.data.text import TinyShakespeare
from src.model import DecoderOnlyTransformer, ModelConfig
from src.train import TrainConfig, train_one_run


def _tiny_model_and_data():
    data = TinyShakespeare(seed=0)
    model_cfg = ModelConfig(
        vocab_size=data.vocab_size, max_seq_len=32,
        d_model=16, n_layers=2, n_heads=2, mlp_ratio=2, dropout=0.0,
    )
    model = DecoderOnlyTransformer(model_cfg)
    return model, data


def test_train_smoke_runs_full_steps_and_val_loss_recorded():
    model, data = _tiny_model_and_data()
    cfg = TrainConfig(
        optimizer_name="adamw", lr=3e-3, batch_size=8, seq_len=16,
        max_steps=30, eval_interval=10, eval_iters=3, seed=0, device="cpu",
    )
    result = train_one_run(model, data, cfg)

    assert result["diverged"] is False
    assert result["steps_completed"] == 30
    assert len(result["val_loss_history"]) == 3  # steps 10, 20, 30
    assert len(result["grad_norm_history"]) == 30
    assert result["best_val_loss"] < float("inf")


def test_train_loss_decreases_from_first_to_last_eval():
    model, data = _tiny_model_and_data()
    cfg = TrainConfig(
        optimizer_name="adamw", lr=5e-3, batch_size=8, seq_len=16,
        max_steps=60, eval_interval=20, eval_iters=5, seed=0, device="cpu",
    )
    result = train_one_run(model, data, cfg)

    first_val = result["val_loss_history"][0]["val_loss"]
    last_val = result["val_loss_history"][-1]["val_loss"]
    assert last_val < first_val


def test_train_detects_divergence_and_stops_early():
    model, data = _tiny_model_and_data()
    cfg = TrainConfig(
        optimizer_name="sgd", lr=1e6, batch_size=8, seq_len=16,
        max_steps=50, eval_interval=10, eval_iters=3, seed=0, device="cpu",
    )
    result = train_one_run(model, data, cfg)

    assert result["diverged"] is True
    assert result["steps_completed"] < cfg.max_steps
