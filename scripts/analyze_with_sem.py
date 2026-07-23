"""Adds standard-error-of-the-mean (SEM = std/sqrt(n), n=3 seeds unless
noted) to the core experiment's plots and tables, across all sweeps run so
far: the original flat 500-step sweep, the schedule ablation, the
longer-training check, and (once available) the hero run. Regenerates the
loss-vs-LR plots with error bars and prints tables with "mean ± SEM" for
each optimizer's own best loss, for pasting into the README.

Caveat, stated once here rather than per-table: with only 3 seeds, SEM
itself is a noisy estimate (SEM's own relative uncertainty is large at
n=3) -- treat these as approximate error bars, not precise confidence
intervals.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = {"adamw": "tab:blue", "sgd": "tab:orange", "nero": "tab:green", "muon": "tab:red"}
OPTS = ["adamw", "sgd", "nero", "muon"]


def summarize(path):
    df = pd.read_csv(path)
    summary = (df.groupby(["optimizer_name", "lr"])
               .agg(mean_best_val=("best_val_loss", "mean"),
                    std_best_val=("best_val_loss", "std"),
                    n_diverged=("diverged", "sum"),
                    n=("diverged", "count"))
               .reset_index())
    summary["sem_best_val"] = summary["std_best_val"] / np.sqrt(summary["n"])
    return df, summary


def own_best_with_sem(summary, opt):
    sub = summary[summary.optimizer_name == opt]
    row = sub.loc[sub.mean_best_val.idxmin()]
    return row.lr, row.mean_best_val, row.sem_best_val


def plot_loss_vs_lr(summary, title, out_path, overlay_summary=None, overlay_label=None,
                     label=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for opt in OPTS:
        sub = summary[summary.optimizer_name == opt].sort_values("lr")
        finite = sub[np.isfinite(sub.mean_best_val)]
        ax.errorbar(finite.lr, finite.mean_best_val, yerr=finite.sem_best_val,
                     marker="o", capsize=3, linestyle="-", color=colors[opt],
                     label=f"{opt}{' (' + label + ')' if label else ''}",
                     alpha=0.9 if overlay_summary is None else 0.55)
        diverged_lrs = sub[sub.n_diverged > 0].lr
        if len(diverged_lrs) > 0:
            ax.scatter(diverged_lrs, [5.0] * len(diverged_lrs), marker="x",
                        color=colors[opt], s=80, zorder=5)
        if overlay_summary is not None:
            osub = overlay_summary[overlay_summary.optimizer_name == opt].sort_values("lr")
            ofinite = osub[np.isfinite(osub.mean_best_val)]
            ax.errorbar(ofinite.lr, ofinite.mean_best_val, yerr=ofinite.sem_best_val,
                         marker="s", capsize=3, linestyle="--", color=colors[opt],
                         label=f"{opt} ({overlay_label})")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("mean best validation loss ± SEM (3 seeds)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(top=3.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


print("=" * 70)
flat_df, flat_summary = summarize("results/core/sweep_results.csv")
sched_df, sched_summary = summarize("results/core/schedule_ablation_results.csv")
long_df, long_summary = summarize("results/core/longer_training_results.csv")

print("\n--- Own best (mean ± SEM), by sweep ---")
for name, summary in [("flat (500 step)", flat_summary), ("schedule (500 step)", sched_summary),
                       ("longer-training (3000 step, top-3 LR only)", long_summary)]:
    print(f"\n[{name}]")
    for opt in OPTS:
        lr, mean, sem = own_best_with_sem(summary, opt)
        print(f"  {opt:6s} {mean:.4f} ± {sem:.4f}  @ lr={lr:g}")

# Regenerate plots with error bars
plot_loss_vs_lr(flat_summary, "Core experiment: loss vs. LR (flat, ±SEM)",
                 "results/core/loss_vs_lr.png")
plot_loss_vs_lr(sched_summary, "Core experiment: flat vs. cosine-warmup schedule (±SEM)",
                 "results/core/schedule_ablation_loss_vs_lr.png",
                 overlay_summary=flat_summary, overlay_label="flat", label="schedule")

if os.path.exists("results/core/hero_sweep_results.csv"):
    hero_df, hero_summary = summarize("results/core/hero_sweep_results.csv")
    print("\n[hero (3000 step, cosine schedule, full grid)]")
    for opt in OPTS:
        lr, mean, sem = own_best_with_sem(hero_summary, opt)
        print(f"  {opt:6s} {mean:.4f} ± {sem:.4f}  @ lr={lr:g}")
    plot_loss_vs_lr(hero_summary, "Core experiment: hero run vs. original 500-step (±SEM)",
                     "results/core/hero_loss_vs_lr.png",
                     overlay_summary=flat_summary, overlay_label="500-step flat", label="hero")

    # Basin width for hero, same methodology as analysis.ipynb
    def is_contiguous(mask):
        idx = np.flatnonzero(mask)
        return len(idx) == 0 or np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))

    def basin_table(summary):
        adamw_best = summary[summary.optimizer_name == "adamw"].mean_best_val.min()
        rows = []
        for X in [0.05, 0.10, 0.20]:
            threshold = adamw_best * (1 + X)
            for opt in OPTS:
                sub = summary[summary.optimizer_name == opt].sort_values("lr").reset_index(drop=True)
                mask = (sub.mean_best_val <= threshold).values
                within = sub[mask]
                if not is_contiguous(mask):
                    print(f"WARNING: non-contiguous qualifying set for {opt} at X={X:.0%}")
                log_span = np.nan if len(within) == 0 else (
                    np.log10(within.lr.max()) - np.log10(within.lr.min()))
                rows.append({"X": X, "optimizer": opt, "own_best": sub.mean_best_val.min(),
                             "n_lrs_within_threshold": len(within), "log10_basin_width": log_span})
        return pd.DataFrame(rows)

    print("\n--- Hero run basin width (X=5/10/20%) ---")
    hb = basin_table(hero_summary)
    print(hb.to_string(index=False))
    hb.to_csv("results/core/hero_basin_width.csv", index=False)
else:
    print("\nresults/core/hero_sweep_results.csv not found yet -- run again once the hero sweep finishes.")
