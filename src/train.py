"""Single training run: model + optimizer + data -> loss history and a
result summary. Used directly (as a function) by sweep.py for each grid
point, and via the CLI below for one-off debugging runs.

Deliberately data/model-agnostic beyond assuming:
  - `data.get_batch(split, batch_size, seq_len, device)` -> (x, y)
  - `model(x, y)` -> (output, loss)
so the same function is reused unchanged for the finance stretch
experiment (different data loader, different model head, same loop).
"""
import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import yaml

from src.optimizers import build_optimizer


def resolve_device(device: str) -> str:
    """'auto' resolves to cuda if available, else cpu; anything else passes
    through unchanged (so an explicit 'cpu' or 'cuda' still works as-is).
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class TrainConfig:
    optimizer_name: str
    lr: float
    batch_size: int = 32
    seq_len: int = 64
    max_steps: int = 500
    eval_interval: int = 50
    eval_iters: int = 20
    seed: int = 0
    device: str = "cpu"
    divergence_threshold: float = 1e4
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def _estimate_val_loss(model, data, cfg: TrainConfig) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = data.get_batch("val", cfg.batch_size, cfg.seq_len, cfg.device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one_run(model: torch.nn.Module, data, cfg: TrainConfig) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    model.to(cfg.device)
    model.train()
    optimizer = build_optimizer(cfg.optimizer_name, model, lr=cfg.lr, **cfg.optimizer_kwargs)

    val_loss_history = []
    grad_norm_history = []
    diverged = False
    steps_completed = 0
    start = time.time()

    for step in range(1, cfg.max_steps + 1):
        x, y = data.get_batch("train", cfg.batch_size, cfg.seq_len, cfg.device)
        optimizer.zero_grad()
        _, loss = model(x, y)

        if not torch.isfinite(loss) or loss.item() > cfg.divergence_threshold:
            diverged = True
            break

        loss.backward()
        total_norm = torch.sqrt(sum(
            p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None
        ))
        grad_norm_history.append(total_norm.item())
        optimizer.step()
        steps_completed = step

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val_loss = _estimate_val_loss(model, data, cfg)
            if not torch.isfinite(torch.tensor(val_loss)):
                diverged = True
                break
            val_loss_history.append({"step": step, "val_loss": val_loss})

    return {
        "optimizer_name": cfg.optimizer_name,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "diverged": diverged,
        "steps_completed": steps_completed,
        "val_loss_history": val_loss_history,
        "best_val_loss": min((h["val_loss"] for h in val_loss_history), default=float("inf")),
        "final_val_loss": val_loss_history[-1]["val_loss"] if val_loss_history else float("inf"),
        "grad_norm_history": grad_norm_history,
        "wall_clock_seconds": time.time() - start,
    }


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    from src.data import build_dataset
    from src.model import DecoderOnlyTransformer, ModelConfig

    data = build_dataset(raw["data"], seed=raw.get("seed", 0))
    model_cfg = ModelConfig(vocab_size=data.vocab_size, **raw["model"])
    model = DecoderOnlyTransformer(model_cfg)
    raw["train"]["device"] = resolve_device(raw["train"].get("device", "cpu"))
    train_cfg = TrainConfig(**raw["train"])

    result = train_one_run(model, data, train_cfg)
    print({k: v for k, v in result.items() if k not in ("val_loss_history", "grad_norm_history")})


if __name__ == "__main__":
    _cli()
