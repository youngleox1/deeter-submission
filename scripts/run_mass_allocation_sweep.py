"""Mass-based automatic LR allocation (Large, Liu et al. NeurIPS'24,
Section 3.3) sweep: fixes the base optimizer (AdamW) and learning rate at
AdamW's own best from the already-completed optimizer_extensions sweep
(lr=7.114e-3), and sweeps `hidden_mass` (relative to input_mass=output_mass=1)
to test whether architecture-based automatic LR allocation helps,
independent of the update-rule question the other sweeps test.

Not run via src/sweep.py since that's built around sweeping LR per
optimizer; hidden_mass is a different axis entirely, so this is a small,
dedicated script reusing train_one_run directly.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.text import TinyShakespeare
from src.model import DecoderOnlyTransformer, ModelConfig
from src.train import TrainConfig, resolve_device, train_one_run

FIXED_LR = 7.114e-3  # AdamW's own best LR from configs/optimizer_extensions_sweep.yaml
HIDDEN_MASSES = [0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = [0, 1, 2]
OUTPUT_CSV = Path("results/optimizer_extensions/mass_allocation_sweep_results.csv")

MODEL_CFG_KWARGS = dict(max_seq_len=128, d_model=128, n_layers=4, n_heads=4, mlp_ratio=4, dropout=0.0)
TRAIN_CFG_KWARGS = dict(batch_size=64, seq_len=128, max_steps=500, eval_interval=100,
                         eval_iters=20, divergence_threshold=10000.0)


def main():
    device = resolve_device("auto")
    results = []

    for hidden_mass in HIDDEN_MASSES:
        for seed in SEEDS:
            data = TinyShakespeare(seed=seed)
            model_cfg = ModelConfig(vocab_size=data.vocab_size, **MODEL_CFG_KWARGS)
            model = DecoderOnlyTransformer(model_cfg)
            cfg = TrainConfig(
                optimizer_name="adamw_mass_alloc", lr=FIXED_LR, seed=seed, device=device,
                optimizer_kwargs=dict(input_mass=1.0, hidden_mass=hidden_mass, output_mass=1.0),
                **TRAIN_CFG_KWARGS,
            )
            result = train_one_run(model, data, cfg)
            result["hidden_mass"] = hidden_mass
            results.append(result)
            print(f"hidden_mass={hidden_mass:<5} seed={seed} "
                  f"diverged={result['diverged']!s:5} best_val={result['best_val_loss']:.4f}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["hidden_mass", "seed", "diverged", "steps_completed", "best_val_loss",
              "final_val_loss", "wall_clock_seconds"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})

    print(f"\nDone: {len(results)} runs, results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
