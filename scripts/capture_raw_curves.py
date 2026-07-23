"""Captures raw val-loss-vs-step training curves (not just the final/best
scalar) at each optimizer's own best LR, under three conditions: the
original flat 500-step run, the 500-step cosine-schedule ablation, and
(once available) the 3000-step hero run. seed=0 only -- these are for
qualitative shape-of-convergence inspection, not a seed-averaged claim.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data import build_dataset
from src.model import DecoderOnlyTransformer, ModelConfig
from src.train import TrainConfig, resolve_device, train_one_run

MODEL_KW = dict(max_seq_len=128, d_model=128, n_layers=4, n_heads=4, mlp_ratio=4, dropout=0.0)
OPT_KWARGS = {
    "adamw": dict(betas=[0.9, 0.95], weight_decay=0.0),
    "sgd": dict(momentum=0.9),
    "nero": dict(beta=0.999),
    "muon": dict(momentum=0.95, adamw_lr=3.0e-3),
}
colors = {"adamw": "tab:blue", "sgd": "tab:orange", "nero": "tab:green", "muon": "tab:red"}


def best_lr_from(csv_path):
    df = pd.read_csv(csv_path)
    summary = (df.groupby(["optimizer_name", "lr"])
               .agg(mean_best_val=("best_val_loss", "mean")).reset_index())
    out = {}
    for opt in ["adamw", "sgd", "nero", "muon"]:
        sub = summary[summary.optimizer_name == opt]
        out[opt] = sub.loc[sub.mean_best_val.idxmin(), "lr"]
    return out


def run_one(opt, lr, max_steps, use_schedule, warmup_steps, eval_interval):
    data = build_dataset({"type": "text"}, seed=0)
    model_cfg = ModelConfig(vocab_size=data.vocab_size, **MODEL_KW)
    model = DecoderOnlyTransformer(model_cfg)
    cfg = TrainConfig(
        optimizer_name=opt, lr=lr, seed=0, batch_size=64, seq_len=128,
        max_steps=max_steps, eval_interval=eval_interval, eval_iters=20,
        device=resolve_device("auto"), optimizer_kwargs=OPT_KWARGS[opt],
        use_cosine_schedule=use_schedule, warmup_steps=warmup_steps, min_lr_ratio=0.1,
    )
    result = train_one_run(model, data, cfg)
    steps = [h["step"] for h in result["val_loss_history"]]
    values = [h["val_loss"] for h in result["val_loss_history"]]
    return steps, values


conditions = [
    ("flat_500", dict(max_steps=500, use_schedule=False, warmup_steps=0, eval_interval=25),
     best_lr_from("results/core/sweep_results.csv"), "-", "o"),
    ("schedule_500", dict(max_steps=500, use_schedule=True, warmup_steps=50, eval_interval=25),
     best_lr_from("results/core/schedule_ablation_results.csv"), "--", "s"),
]
if os.path.exists("results/core/hero_sweep_results.csv"):
    conditions.append((
        "hero_3000", dict(max_steps=3000, use_schedule=True, warmup_steps=500, eval_interval=150),
        best_lr_from("results/core/hero_sweep_results.csv"), ":", "^"))
else:
    print("results/core/hero_sweep_results.csv not found yet -- skipping hero curves this pass.")

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
curve_rows = []
for ax, opt in zip(axes.flat, ["adamw", "sgd", "nero", "muon"]):
    for label, kw, best_lrs, linestyle, marker in conditions:
        lr = best_lrs[opt]
        steps, history = run_one(opt, lr, kw["max_steps"], kw["use_schedule"],
                                  kw["warmup_steps"], kw["eval_interval"])
        ax.plot(steps, history, linestyle=linestyle, marker=marker, markersize=3,
                 color=colors[opt], alpha=0.9 if label == "hero_3000" else 0.6,
                 label=f"{label} (lr={lr:g})")
        for s, v in zip(steps, history):
            curve_rows.append({"optimizer": opt, "condition": label, "lr": lr, "step": s, "val_loss": v})
        print(f"{opt:6s} {label:12s} lr={lr:g} final_val={history[-1]:.4f}")
    ax.set_title(opt)
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.legend(fontsize=7)
fig.suptitle("Raw val-loss-vs-step curves at each optimizer's own best LR (seed=0)")
fig.tight_layout()
fig.savefig("results/core/raw_curves.png", dpi=150)
pd.DataFrame(curve_rows).to_csv("results/core/raw_curves.csv", index=False)
print("\nSaved results/core/raw_curves.png and results/core/raw_curves.csv")
