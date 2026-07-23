"""Summary figure for the top of Results: loss vs. LR normalized to each
optimizer's own best LR (x = log10(lr / own_best_lr)), so all four curves
are aligned at their own optimum regardless of where that optimum falls in
absolute LR terms. Two panels: 500-step flat LR (left) vs. the 3000-step
cosine-schedule hero run (right), sharing y-axis for a direct read on how
training length + schedule changes both best loss and basin shape.
"""
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
    return summary


def plot_panel(ax, summary, title):
    for opt in OPTS:
        sub = summary[summary.optimizer_name == opt].sort_values("lr")
        finite = sub[np.isfinite(sub.mean_best_val)]
        best_lr = finite.loc[finite.mean_best_val.idxmin(), "lr"]
        x = np.log10(finite.lr / best_lr)
        ax.errorbar(x, finite.mean_best_val, yerr=finite.sem_best_val,
                     marker="o", capsize=3, color=colors[opt], label=opt)
        diverged = sub[sub.n_diverged > 0]
        if len(diverged) > 0:
            xd = np.log10(diverged.lr / best_lr)
            ax.scatter(xd, [ax_ylim_top] * len(xd), marker="x", color=colors[opt], s=70, zorder=5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("log10(LR / optimizer's own best LR)")
    ax.set_title(title)


ax_ylim_top = 2.6  # shared marker height for divergence x's, set before plotting

flat_summary = summarize("results/core/sweep_results.csv")
hero_summary = summarize("results/core/hero_sweep_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
plot_panel(axes[0], flat_summary, "500-step, flat LR")
plot_panel(axes[1], hero_summary, "3000-step, cosine schedule (hero)")
axes[0].set_ylabel("mean best validation loss ± SEM (3 seeds)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(1.4, ax_ylim_top)
fig.suptitle("Loss vs. LR, aligned to each optimizer's own best LR (x=0)")
fig.tight_layout()
fig.savefig("results/core/summary_aligned_lr.png", dpi=150)
print("Saved results/core/summary_aligned_lr.png")

# Quantitative summary: basin "sharpness" via width @ X=10% around each
# optimizer's OWN best (not AdamW-anchored) -- directly answers "how does
# longer training change basin size" per optimizer, not just relative to AdamW.
def own_basin_width(summary, X=0.10):
    rows = []
    for opt in OPTS:
        sub = summary[summary.optimizer_name == opt].sort_values("lr").reset_index(drop=True)
        own_best = sub.mean_best_val.min()
        threshold = own_best * (1 + X)
        mask = (sub.mean_best_val <= threshold).values
        within = sub[mask]
        width = np.nan if len(within) == 0 else np.log10(within.lr.max()) - np.log10(within.lr.min())
        rows.append({"optimizer": opt, "own_best": own_best, "log10_basin_width_own_X10": width})
    return pd.DataFrame(rows)

print("\n--- Basin width around EACH OPTIMIZER'S OWN best (X=10%), flat vs. hero ---")
flat_own = own_basin_width(flat_summary).set_index("optimizer")
hero_own = own_basin_width(hero_summary).set_index("optimizer")
cmp = flat_own.join(hero_own, lsuffix="_flat", rsuffix="_hero")
print(cmp.to_string())
