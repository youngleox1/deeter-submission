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


def test_cosine_schedule_disabled_by_default_keeps_lr_flat():
    """Backward-compatibility guard: every prior sweep's results assumed a
    flat LR. use_cosine_schedule defaults to False, so this must hold
    unless a config explicitly opts in.
    """
    model, data = _tiny_model_and_data()
    cfg = TrainConfig(
        optimizer_name="adamw", lr=3e-3, batch_size=8, seq_len=16,
        max_steps=20, eval_interval=10, eval_iters=2, seed=0, device="cpu",
    )
    result = train_one_run(model, data, cfg)

    assert all(lr == 3e-3 for lr in result["lr_history"])


def test_cosine_schedule_enabled_follows_warmup_then_decay():
    model, data = _tiny_model_and_data()
    peak_lr = 3e-3
    cfg = TrainConfig(
        optimizer_name="adamw", lr=peak_lr, batch_size=8, seq_len=16,
        max_steps=20, eval_interval=10, eval_iters=2, seed=0, device="cpu",
        use_cosine_schedule=True, warmup_steps=5, min_lr_ratio=0.1,
    )
    result = train_one_run(model, data, cfg)
    lrs = result["lr_history"]

    assert len(lrs) == 20
    # warmup: step 1 -> peak/5, ramping up to step 5 -> peak
    assert abs(lrs[0] - peak_lr / 5) < 1e-9
    assert abs(lrs[4] - peak_lr) < 1e-9
    # decay phase: strictly decreasing after warmup, ending above the floor
    assert lrs[4] > lrs[10] > lrs[19]
    assert lrs[19] >= peak_lr * 0.1 - 1e-9
