"""Measures, rather than just asserts, each optimizer's per-step compute
cost and memory footprint on the actual core-experiment model
(configs/core_sweep.yaml's architecture). Two things are reported:

1. Optimizer STATE memory (exact, analytic): total elements across every
   tensor in optimizer.state, as a multiple of the model's own parameter
   count. This isolates the optimizer's own footprint from batch-size- or
   activation-dependent effects.
2. Empirical per-step wall-clock time (GPU, synchronized) and peak CUDA
   memory during a short run, for a real, not just theoretical, comparison.

Also cross-checks (1)/(2) against results/core/sweep_results.csv's own
recorded wall_clock_seconds/steps_completed, since that data already
exists from the full sweep and should roughly agree.
"""
import time
import numpy as np
import pandas as pd
import torch

from src.data import build_dataset
from src.model import DecoderOnlyTransformer, ModelConfig
from src.optimizers import build_optimizer

MODEL_KW = dict(max_seq_len=128, d_model=128, n_layers=4, n_heads=4, mlp_ratio=4, dropout=0.0)
OPT_KWARGS = {
    "adamw": dict(betas=[0.9, 0.95], weight_decay=0.0),
    "sgd": dict(momentum=0.9),
    "nero": dict(beta=0.999),
    "muon": dict(momentum=0.95, adamw_lr=3.0e-3),
}
# Each optimizer's own best LR from results/core/sweep_results.csv
BEST_LR = {"adamw": 0.007114, "sgd": 0.3, "nero": 0.03, "muon": 0.02}

device = "cuda" if torch.cuda.is_available() else "cpu"
data = build_dataset({"type": "text"}, seed=0)

print(f"device={device}\n")
rows = []
for opt_name in ["adamw", "sgd", "nero", "muon"]:
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(ModelConfig(vocab_size=data.vocab_size, **MODEL_KW)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(opt_name, model, lr=BEST_LR[opt_name], **OPT_KWARGS[opt_name])

    # Warmup step so optimizer.state is populated (state dicts are built
    # lazily on first .step() for torch's own AdamW/SGD; Nero builds state
    # at construction; Muon lazily too).
    x, y = data.get_batch("train", 64, 128, device)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state_elems = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                state_elems += v.numel()
    # torch SGD/AdamW nest extra bookkeeping (e.g. 'step' as a tensor in
    # newer torch versions) -- doesn't materially change the picture but
    # note it's included above as-is, not hand-filtered.

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    n_steps = 50
    start = time.time()
    for _ in range(n_steps):
        x, y = data.get_batch("train", 64, 128, device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    ms_per_step = 1000 * elapsed / n_steps
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")

    rows.append({
        "optimizer": opt_name, "n_params": n_params, "state_elems": state_elems,
        "state_ratio": state_elems / n_params, "state_mb_fp32": state_elems * 4 / 1e6,
        "ms_per_step": ms_per_step, "peak_cuda_mb": peak_mb,
    })
    print(f"{opt_name:6s} params={n_params:,}  state_elems={state_elems:,} "
          f"({state_elems / n_params:.2f}x params, {state_elems * 4 / 1e6:.2f} MB fp32)  "
          f"{ms_per_step:.2f} ms/step  peak_cuda={peak_mb:.1f} MB")

df = pd.DataFrame(rows)
df.to_csv("results/core/optimizer_cost.csv", index=False)
print("\nSaved results/core/optimizer_cost.csv")

print("\n--- Cross-check against the full sweep's own recorded timings ---")
sweep = pd.read_csv("results/core/sweep_results.csv")
sweep["ms_per_step"] = 1000 * sweep.wall_clock_seconds / sweep.steps_completed
print(sweep.groupby("optimizer_name").ms_per_step.agg(["mean", "std"]).to_string())
