# Does architecture-aware optimization win a wider learning-rate basin?

## Motivation

Optimal learning rate is known to shift with model size and architecture
rather than staying fixed (Kaplan et al., 2020, *"Scaling Laws for Neural
Language Models"*), which is precisely why LR search is a recurring cost
rather than a one-time one — each new scale or dataset can, in principle,
need its own sweep. A wider basin of near-optimal learning rates would help
in a few concrete, distinct ways:

- **Fewer sweep points needed, illustratively.** This is a hypothetical
  arithmetic example, not an established empirical result: *if* AdamW needs
  ~5-7 LR values on a log grid to find a near-optimal setting, and *if* an
  architecture-aware method has, say, a 3x wider basin at the same loss
  tolerance, that *could* translate into needing fewer grid points to reach
  similar confidence in the chosen LR. Whether that relationship actually
  holds, and by how much, is exactly what the core experiment below checks —
  it is a hypothesis to test, not a claimed result.
- **LR transfer across scale.** This is the practical motivation behind μP
  (Yang & Hu et al., 2021, *"Tensor Programs V: Tuning Large Neural Networks
  via Zero-Shot Hyperparameter Transfer"*) and modular-norm-style work
  (Large, Liu et al., NeurIPS 2024, *"Scalable Optimization in the Modular
  Norm"* — prior work of the author of this repo): tune once at small scale,
  reuse the same LR at larger scale instead of re-sweeping per size. A wider
  basin is what would make that transfer more forgiving of imperfect scaling
  rules.
- **Avoiding catastrophic divergence, not just suboptimal loss.** Wortsman et
  al. (2023, DeepMind, *"Small-Scale Proxies for Large-Scale Transformer
  Training Instabilities"*) show that small-scale LR sweeps can be used to
  anticipate and avoid training instabilities that would otherwise only
  appear at large scale. This is the concrete real-world case for the
  divergence-rate metric defined below: a wider *stable* region, not just a
  wider *near-optimal* region, is what actually de-risks a large training
  run in practice.
- **Lower operational risk under periodic retraining (own reasoning, not a
  literature claim).** If a model is refit on a regular cadence as the
  underlying data distribution drifts — the finance stretch experiment below
  is a concrete instance of this — a wide-basin optimizer plausibly needs
  less frequent LR re-sweeping. This is a speculative extension motivating
  the stretch experiment, not a result established elsewhere, and is treated
  as such.

## Hypothesis

An architecture-aware optimizer (update scaling based on layer shape/role, in the
style of modular-norm / architecture-aware LR scaling) does not just match a
well-tuned AdamW baseline at its best learning rate — it stays close to its best
loss across a *wider range* of learning rates. In other words: it trades a small
amount of best-case performance for robustness to LR misspecification.

This is tested twice, on two structurally different sequence domains:

1. **Core experiment**: small decoder-only transformer, language modeling on a
   small public text corpus.
2. **Stretch experiment**: the same model/optimizer code, applied to next-day
   return/direction forecasting on public daily equity data (via `yfinance`).

### Why decoder-only transformer on text, not vision (tiny ViT on MNIST/CIFAR)?

Both are legitimate choices, and both have precedent in prior architecture-aware
optimizer work (vision CNNs in the original Nero paper; GPT-style transformers
in modular-norm work). Vision would be the faster, more precedented choice if
this were a standalone core experiment. The reason for text+decoder here
specifically: the stretch experiment is also sequence forecasting, so keeping
the **model family fixed** (decoder-only sequence transformer) across core and
stretch means only the *data domain* changes between the two experiments, not
the domain and the architecture family at once. That keeps the transfer claim
("does the basin-width finding hold in a new domain") interpretable, rather
than conflated with "does it also hold for a different architecture class."

## Optimizer candidates

| Optimizer | Role | Architecture-aware? |
|---|---|---|
| AdamW | Standard baseline | No |
| SGD + momentum | Classic non-adaptive baseline | No |
| Nero | Own prior work (ICML'21, "Learning by Turning") | Yes — neuron-wise normalized updates |
| Muon | Current (2024-2025) architecture-aware method, orthogonalized momentum update, closely related in spirit to modular-norm | Yes — applied to 2D hidden-layer matrices |

Scoped to four rather than including LAMB or a full Modula/modular-norm
implementation — both are reasonable additions but carry more implementation
risk than their marginal signal justifies in this time budget. Noted
explicitly in Limitations as deliberate scope cuts, not oversights.

## Success criteria (declared before running any experiment)

Two complementary metrics are reported, not one — basin width alone hides the
more decision-relevant failure mode (divergence), and divergence alone doesn't
capture near-threshold sensitivity.

**Basin width.** For a given domain, an optimizer "wins" on this axis if,
across a fixed grid of learning rates:

- Its best achieved validation loss is within **X%** of AdamW's best validation
  loss (i.e. it is not simply *better* by finding a different optimum — parity
  at best-LR is the null-hypothesis-compatible outcome), **and**
- The range of learning rates for which it stays within that same X% band is
  **at least 2x wider** (in log-LR space) than AdamW's corresponding band.

**Divergence rate.** Fraction of (LR, seed) runs that diverge (NaN/Inf loss,
or loss exceeding a fixed blowup threshold) at each LR. This is reported
alongside basin width because it is binary and unambiguous, and arguably the
more practically important reason to want an architecture-aware optimizer —
avoiding catastrophic runs, not just staying within some tolerance band.

If either optimizer fails to win on either axis, that is reported as a
negative result, not re-framed after the fact. (Exact value of X% and the
blowup threshold are fixed in the core experiment config before the sweep is
run, and are not tuned post-hoc.)

**Secondary metrics** (diagnostic, not headline):
- Compute-to-target: steps needed to reach a fixed validation loss threshold,
  at each optimizer's own best LR (sample efficiency).
- Cross-seed variance at fixed LR (a distinct notion of robustness —
  run-to-run consistency rather than LR sensitivity).
- Update-norm trajectories over training, as a mechanistic diagnostic for
  *why* an optimizer is more or less robust (not a pass/fail criterion).

For the finance stretch, additionally reported:
- Directional accuracy vs. a naive baseline (predict-no-change /
  predict-previous-direction).
- A calibration metric (Brier score) rather than accuracy alone — a
  well-calibrated but modest model is a more honest result than a bare
  accuracy number, and calibration is the more quant-relevant property.

No claim of trading edge, Sharpe ratio, or backtest performance is made — the
model and task are deliberately simple, intended only to test whether the
basin-width/divergence-rate findings transfer out of domain.

## References

- Kaplan, J. et al. "Scaling Laws for Neural Language Models." 2020.
- Yang, G. & Hu, E. et al. "Tensor Programs V: Tuning Large Neural Networks
  via Zero-Shot Hyperparameter Transfer." 2021.
- Liu, Y., Bernstein, J., Meister, M., Yue, Y. "Learning by Turning: Neural
  Architecture-Aware Optimisation." ICML 2021.
- Large, T., Liu, Y. et al. "Scalable Optimization in the Modular Norm."
  NeurIPS 2024.
- Wortsman, M. et al. "Small-Scale Proxies for Large-Scale Transformer
  Training Instabilities." 2023.
- Jordan, K. et al. "Muon: An optimizer for hidden layers in neural
  networks." 2024. (Technical report / blog, not peer-reviewed — noted as
  such since it's a newer, less formally vetted method than the others
  above.)
- You, Y. et al. "Large Batch Optimization for Deep Learning: Training BERT
  in 76 Minutes." ICLR 2020. (LAMB — related method, not included; see
  Optimizer candidates above.)

## Data

- Core: [TBD small public text corpus — filled in once chosen]
- Finance: public daily OHLCV data pulled via the `yfinance` package for a
  small, fixed list of liquid US equity tickers. This is external,
  freely available market data; no proprietary or non-public data is used.
  See `src/data/finance.py` for the exact fetch logic and ticker list.

## Repo layout

```
src/
  model.py          small decoder-only transformer
  optimizers.py     AdamW baseline + architecture-aware variants
  train.py          single training run
  sweep.py          LR x optimizer x seed grid driver
  data/
    text.py         core experiment data loader
    finance.py       finance stretch data loader
configs/            YAML configs for core and finance sweeps
results/             sweep outputs (csv) and generated plots
tests/               unit/smoke tests (see below)
analysis.ipynb       generates all plots from results/*.csv
scripts/run_all.sh   reproduce everything end to end
```

## How to run

```bash
pip install -r requirements.txt
pytest                          # run test suite
python -m src.sweep --config configs/core_sweep.yaml
python -m src.sweep --config configs/finance_sweep.yaml
```

## Status

This README is being filled in incrementally as the project progresses — see
git history for how the experiment evolved. Results sections below are added
once the corresponding sweeps have actually been run.

### Core experiment results

Full sweep: 4 optimizers x 9 LRs x 3 seeds = 108 runs, 500 steps each. Raw
results: `results/core/sweep_results.csv`. Analysis and plots:
`analysis.ipynb` (see `results/core/*.png` for the rendered figures).

**Process note, disclosed rather than fixed quietly:** the loss-tolerance
threshold X% (used to define "within X% of AdamW's best") was never actually
committed to a config file before this sweep ran, despite the intent stated
above. All three plausible values (5%, 10%, 20%) are reported below rather
than picking one after seeing the results. The qualitative conclusion is
identical across all three, so the finding is not an artifact of this gap —
but the gap itself is real and is reported as such, not glossed over.

| Optimizer | Own best val loss | Beats AdamW's best (1.718)? | log10 basin width @ X=10% |
|---|---|---|---|
| AdamW | 1.718 | (reference) | 0.375 |
| **Muon** | **1.654** | **Yes** | **1.125 (3x wider)** |
| SGD | 2.170 | No — never within 20% at any LR tested | diverges above lr≈0.71 (all seeds) |
| Nero | 2.058 | No — within 20% at exactly one LR (0.03), zero-width basin | n/a |

**Revision note:** an earlier version of this table used a meaningfully
buggy Nero implementation (see `v0.5.0-nero-fix` tag / git history for the
full diagnosis — no mean-centering, wrong re-projection target, sum-vs-mean
second moment, spurious momentum on 1D params). Nero's numbers above are
from the corrected implementation, matching the reference
(github.com/jxbz/nero) closely. AdamW/SGD/Muon are unchanged (confirmed
identical to prior digits — they weren't touched by the fix, and results
are seeded/deterministic).

Two separate findings, not one clean story:

1. **Muon wins on both pre-declared axes, and robustly** — holds at X=5%,
   10%, and 20% alike. It also beats AdamW's best loss outright, which is
   actually outside what the pre-declared criteria anticipated (the design
   assumed an architecture-aware method would trade a bit of best-case
   performance for a wider basin; Muon didn't have to make that trade here).
2. **The corrected Nero still underperforms AdamW substantially**, though
   less than the buggy version did (best loss 2.058 vs the old 2.091; best
   LR shifted a full decade, from 0.0023 to 0.03). At the loosest tested
   threshold (X=20%) it now barely qualifies as "within range" — but at a
   single LR point, not a range, so it has effectively zero basin width by
   this metric regardless of implementation correctness. This is a genuine
   negative result, not a bug (Nero's tests now check the correct
   invariants — unit-norm + mean-zero per neuron after construction and
   after every step — and all pass). Plausible explanations for the
   remaining gap, none confirmed: no LR warmup/schedule is used here, and
   Nero's reported benefits may show up more at larger scale or longer
   training than this 500-step toy setting. Reported as an open question,
   not explained away.

**Divergence rate:** SGD is the only optimizer that diverges anywhere in its
sampled grid (all 3 seeds, for every LR ≥ 0.71 — see
`results/core/divergence_rate.png`). AdamW, Nero, and Muon never diverge
anywhere in their sampled ranges. This partially, not fully, supports the
architecture-aware motivation: it cleanly separates SGD from the other
three, but AdamW (not architecture-aware, by this project's definition)
also never diverges — so divergence avoidance alone doesn't cleanly
distinguish "architecture-aware" from "not" in this experiment.

### Ablation: does removing LayerNorm's affine params help Nero?

Raised during review: Nero's projection assumes a neuron's weight scale
(and, per the corrected implementation, its mean) is irrelevant because
downstream normalization absorbs it — an assumption that weakens if the
normalization layer itself has a learnable affine scale/shift (handled by
a separate, uncoordinated update rule for 1D params). Tested directly:
same LR grid/seeds/steps as the core sweep, `model.layernorm_affine=False`
(`configs/ablation_nero_no_ln_affine.yaml`,
`results/ablations/nero_no_ln_affine/sweep_results.csv`), rerun against the
corrected Nero.

**Result: mixed, not a clean confirmation or rejection.** At most LRs
(0.0009 through 0.03, and again at 0.07), keeping the affine params does
slightly better than removing them — the hypothesis is not supported
there. But at lr=0.4, removing them is clearly better (2.85 vs 3.08), and
at the highest LR (0.95), keeping them is much better (3.28 vs 5.71 —
removing affine makes the high-LR degradation much worse, not better).
So the honest read is: LayerNorm's affine params don't appear to be the
main driver of Nero's overall underperformance (removing them doesn't
fix it, and mostly makes things slightly worse) — but they do appear to
provide some stabilizing effect specifically at high LR, which is the
opposite direction from the original hypothesis. Reported as a
tested-and-not-confirmed hypothesis with a genuine, unanticipated
secondary finding, not retro-fitted into either story after the fact.

### Finance stretch results
_TBD — sweep rerunning with corrected Nero; see `finance-stretch` branch
for the in-progress writeup and a separate investigation into whether any
learnable signal exists in this data at all._

## Limitations

- **X% was not pre-registered in a config file before the core sweep ran**
  (see Core experiment results above for how this is handled: all three
  candidate values are reported, and the conclusion is stable across them).
- **No LR schedule of any kind is used anywhere in this project** — every
  run is a flat, constant LR for its full duration, no warmup, no decay.
  This is a real limitation, not specific to any one optimizer: it likely
  understates AdamW's usable LR range in particular (warmup commonly
  prevents the kind of early instability that would show up here as poor
  high-LR performance), and real-world Muon usage is essentially always
  paired with a decay schedule, so testing it constant-LR-only is a
  departure from how it's normally deployed. Raised during review; not
  addressed in this submission due to time, but a natural next step would
  be one additional ablation (single cosine+warmup config, same LR grid)
  rather than expanding the main grid, to avoid reopening the same
  "which schedule is fair to all four" problem X% already ran into.
- **500 training steps is short** relative to typical conventions for this
  exact toy setup (e.g. nanoGPT's own char-level tiny-Shakespeare example
  commonly trains several thousand steps). This means the core experiment's
  ranking reflects early-training relative behavior, not asymptotic
  quality — optimizers with fast early descent are structurally favored
  over ones whose properties play out over a longer horizon. Not addressed
  here due to time; a longer-training check on the top 2-3 LRs per
  optimizer would be the natural follow-up.
- **Muon is a simplified, from-scratch reproduction**, not a verbatim port
  of the reference implementation — see `src/optimizers.py` docstring for
  exactly what's simplified. **Nero was initially a much more meaningfully
  different reimplementation** (missing mean-centering, wrong projection
  target, sum-vs-mean second moment, spurious momentum on 1D params) until
  corrected against the actual reference code (github.com/jxbz/nero) —
  see `v0.5.0-nero-fix` tag. The corrected version's remaining
  underperformance vs. AdamW is treated as a genuine finding, not
  attributed to further unverified implementation differences.
- **LAMB and full Modula/modular-norm were scoped out** of the optimizer
  comparison (see Optimizer candidates above) for time-budget reasons, not
  because they're less relevant.
- Muon's per-parameter-group hybrid design (orthogonalized update for 2D
  hidden matrices, AdamW-style fallback for the rest) means its "LR" in the
  sweep only varies the Muon branch; the fallback-branch LR is held fixed
  (see `configs/core_sweep.yaml`). A full 2D sweep over both would be more
  thorough but was out of scope here.
