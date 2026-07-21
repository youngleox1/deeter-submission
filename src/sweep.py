"""LR x optimizer x seed grid driver.

Each optimizer sweeps its own LR grid (see configs/core_sweep.yaml), not a
shared grid across all four -- AdamW/SGD/Nero/Muon operate at very
different natural LR scales, so a single shared grid would under-sample
around whichever optimizer's true optimum falls outside that range and
bias the basin-width comparison this whole experiment is trying to make.
"""
import argparse
import csv
import itertools
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.data.text import TinyShakespeare
from src.model import DecoderOnlyTransformer, ModelConfig
from src.train import TrainConfig, resolve_device, train_one_run

CSV_FIELDS = [
    "optimizer_name", "lr", "seed", "diverged", "steps_completed",
    "best_val_loss", "final_val_loss", "mean_grad_norm", "wall_clock_seconds",
]


def run_sweep(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_cfg_kwargs = config["model"]
    train_cfg_kwargs = dict(config["train"])
    train_cfg_kwargs["device"] = resolve_device(train_cfg_kwargs.get("device", "cpu"))
    seeds = config["sweep"]["seeds"]
    lr_grids = config["sweep"]["lr_grids"]
    optimizer_kwargs_by_name = config["sweep"].get("optimizer_kwargs", {})

    results = []
    for optimizer_name, lrs in lr_grids.items():
        for lr, seed in itertools.product(lrs, seeds):
            data = TinyShakespeare(seed=seed)
            model_cfg = ModelConfig(vocab_size=data.vocab_size, **model_cfg_kwargs)
            model = DecoderOnlyTransformer(model_cfg)
            cfg = TrainConfig(
                optimizer_name=optimizer_name, lr=lr, seed=seed,
                optimizer_kwargs=optimizer_kwargs_by_name.get(optimizer_name, {}),
                **train_cfg_kwargs,
            )
            result = train_one_run(model, data, cfg)
            results.append(result)
            print(f"{optimizer_name:8s} lr={lr:<12g} seed={seed} "
                  f"diverged={result['diverged']!s:5} best_val={result['best_val_loss']:.4f}")
    return results


def save_results_csv(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            grad_norms = r["grad_norm_history"]
            mean_gn = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
            writer.writerow({
                "optimizer_name": r["optimizer_name"],
                "lr": r["lr"],
                "seed": r["seed"],
                "diverged": r["diverged"],
                "steps_completed": r["steps_completed"],
                "best_val_loss": r["best_val_loss"],
                "final_val_loss": r["final_val_loss"],
                "mean_grad_norm": mean_gn,
                "wall_clock_seconds": r["wall_clock_seconds"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = run_sweep(config)
    save_results_csv(results, Path(config["output_csv"]))
    n_diverged = sum(r["diverged"] for r in results)
    print(f"\nDone: {len(results)} runs, {n_diverged} diverged, "
          f"results written to {config['output_csv']}")


if __name__ == "__main__":
    main()
