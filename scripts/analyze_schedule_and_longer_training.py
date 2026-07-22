"""One-off analysis script for the two follow-up sweeps (schedule ablation,
longer-training check) run to address the two Limitations bullets about
missing LR schedule and short training. Prints the same basin-width
methodology used in analysis.ipynb (see that notebook for the full
explanation) applied to the new data, plus flat-vs-schedule and
short-vs-long comparisons, and saves two comparison plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = {"adamw": "tab:blue", "sgd": "tab:orange", "nero": "tab:green", "muon": "tab:red"}


def is_contiguous(mask):
    idx = np.flatnonzero(mask)
    return len(idx) == 0 or np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))


def basin_table(summary, label):
    adamw_best = summary[summary.optimizer_name == "adamw"].mean_best_val.min()
    print(f"\n[{label}] AdamW best (reference): {adamw_best:.4f}")
    rows = []
    for X in [0.05, 0.10, 0.20]:
        threshold = adamw_best * (1 + X)
        for opt in ["adamw", "sgd", "nero", "muon"]:
            sub = summary[summary.optimizer_name == opt].sort_values("lr").reset_index(drop=True)
            mask = (sub.mean_best_val <= threshold).values
            within = sub[mask]
            contiguous = is_contiguous(mask)
            if not contiguous:
                print(f"WARNING: non-contiguous qualifying set for {opt} at X={X:.0%}")
            log_span = np.nan if len(within) == 0 else (
                np.log10(within.lr.max()) - np.log10(within.lr.min()))
            rows.append({"X": X, "optimizer": opt, "own_best": sub.mean_best_val.min(),
                         "n_lrs_within_threshold": len(within), "log10_basin_width": log_span})
    return pd.DataFrame(rows)


def summarize(path):
    df = pd.read_csv(path)
    summary = (df.groupby(["optimizer_name", "lr"])
               .agg(mean_best_val=("best_val_loss", "mean"),
                    n_diverged=("diverged", "sum"), n=("diverged", "count"))
               .reset_index())
    return df, summary


print("=" * 70)
flat_df, flat_summary = summarize("results/core/sweep_results.csv")
sched_df, sched_summary = summarize("results/core/schedule_ablation_results.csv")

print("\n--- Per-optimizer best (mean across seeds), flat vs. schedule ---")
for opt in ["adamw", "sgd", "nero", "muon"]:
    f = flat_summary[flat_summary.optimizer_name == opt]
    s = sched_summary[sched_summary.optimizer_name == opt]
    f_best_row = f.loc[f.mean_best_val.idxmin()]
    s_best_row = s.loc[s.mean_best_val.idxmin()]
    f_diverged = int(f.n_diverged.sum())
    s_diverged = int(s.n_diverged.sum())
    print(f"{opt:6s} flat:   best={f_best_row.mean_best_val:.4f} @ lr={f_best_row.lr:<10g} "
          f"diverged={f_diverged}/{int(f.n.sum())}")
    print(f"{opt:6s} sched:  best={s_best_row.mean_best_val:.4f} @ lr={s_best_row.lr:<10g} "
          f"diverged={s_diverged}/{int(s.n.sum())}")

flat_basin = basin_table(flat_summary, "flat (original)")
sched_basin = basin_table(sched_summary, "schedule ablation")

print("\n--- log10 basin width, flat vs. schedule, by X ---")
merged = flat_basin.merge(sched_basin, on=["X", "optimizer"], suffixes=("_flat", "_sched"))
print(merged[["X", "optimizer", "log10_basin_width_flat", "log10_basin_width_sched"]]
      .to_string(index=False))

# Overlay plot: flat (solid) vs. schedule (dashed)
fig, ax = plt.subplots(figsize=(7, 5))
for opt in ["adamw", "sgd", "nero", "muon"]:
    fs = flat_summary[flat_summary.optimizer_name == opt].sort_values("lr")
    ss = sched_summary[sched_summary.optimizer_name == opt].sort_values("lr")
    fs_finite = fs[np.isfinite(fs.mean_best_val)]
    ss_finite = ss[np.isfinite(ss.mean_best_val)]
    ax.plot(fs_finite.lr, fs_finite.mean_best_val, marker="o", linestyle="-",
             color=colors[opt], label=f"{opt} (flat)", alpha=0.55)
    ax.plot(ss_finite.lr, ss_finite.mean_best_val, marker="s", linestyle="--",
             color=colors[opt], label=f"{opt} (cosine+warmup)")
ax.set_xscale("log")
ax.set_xlabel("learning rate (log scale)")
ax.set_ylabel("mean best validation loss (across seeds)")
ax.set_title("Core experiment: flat LR vs. cosine-warmup schedule")
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(top=3.5)
fig.tight_layout()
fig.savefig("results/core/schedule_ablation_loss_vs_lr.png", dpi=150)
print("\nSaved results/core/schedule_ablation_loss_vs_lr.png")

print("\n" + "=" * 70)
print("--- Longer-training check (top-3 LRs/optimizer, 3000 vs. 500 steps) ---")
long_df = pd.read_csv("results/core/longer_training_results.csv")
long_summary = (long_df.groupby(["optimizer_name", "lr"])
                .agg(mean_best_val=("best_val_loss", "mean"),
                     n_diverged=("diverged", "sum"), n=("diverged", "count"))
                .reset_index())
rows = []
for opt in ["adamw", "sgd", "nero", "muon"]:
    l = long_summary[long_summary.optimizer_name == opt]
    best_row = l.loc[l.mean_best_val.idxmin()]
    rows.append((opt, best_row.lr, best_row.mean_best_val))
rows.sort(key=lambda r: r[2])
print("Ranking at 3000 steps (best to worst), among each optimizer's own top-3 500-step LRs:")
for rank, (opt, lr, loss) in enumerate(rows, 1):
    print(f"  {rank}. {opt:6s} best={loss:.4f} @ lr={lr:g}")

print("\nFor comparison, 500-step ranking (best to worst):")
flat_ranked = sorted(
    [(opt, flat_summary[flat_summary.optimizer_name == opt].mean_best_val.min()) for opt in
     ["adamw", "sgd", "nero", "muon"]], key=lambda r: r[1])
for rank, (opt, loss) in enumerate(flat_ranked, 1):
    print(f"  {rank}. {opt:6s} best={loss:.4f}")
