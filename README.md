# Does architecture-aware optimization win a wider learning-rate basin?

## TL;DR

- **Question:** does an architecture-aware optimizer (Nero, Muon) trade a
  bit of best-case loss for a *wider* range of learning rates that stay
  near-optimal, compared to AdamW/SGD? Tested on a small decoder-only
  transformer, on two domains: char-level text and discretized daily
  equity returns.
- **Core result (text):** **Muon wins outright** — best loss *and* a 3x
  wider basin than AdamW, robust across three different tolerance
  thresholds. **Nero underperforms** even after a from-scratch
  reimplementation bug was caught and fixed against the reference code —
  a genuine open question, not explained away. SGD is the only optimizer
  that diverges anywhere in its sampled range.
- **Finance stretch:** the neural pipeline finds ~no signal (near
  uniform-baseline loss, no directional edge over a naive baseline) — but
  a simple 5-lag logistic regression on the *same data* clearly does
  (beats naive on all 6 tickers). So the honest conclusion isn't "market
  efficiency," it's "this pipeline's 8-bin discretization + cross-entropy
  objective fails to extract a signal that demonstrably exists." See
  [Branches](#branches) — that finding lives on `finance-stretch`.
- Full navigation: [Table of contents](#table-of-contents).

## Table of contents

- [Branches](#branches)
- [Motivation](#motivation)
- [Hypothesis](#hypothesis)
- [Optimizer candidates](#optimizer-candidates)
- [Success criteria](#success-criteria-declared-before-running-any-experiment)
- [Data](#data)
- [Repo layout](#repo-layout)
- [How to run](#how-to-run)
- [Results](#results)
  - [Core experiment results](#core-experiment-results)
  - [Ablation: LayerNorm affine params](#ablation-does-removing-layernorms-affine-params-help-nero)
  - [Finance stretch results](#finance-stretch-results)
- [Limitations](#limitations) *(folded)*
- [References](#references) *(folded)*

## Branches

This repo uses three branches rather than one long-running one — each is a
self-contained unit of work with its own commits/results/writeup:

| Branch | Contents |
|---|---|
| [`master`](https://github.com/youngleox1/deeter-submission/tree/master) | Core text experiment (this README), the Nero rewrite, and the LayerNorm-affine ablation. |
| [`finance-stretch`](https://github.com/youngleox1/deeter-submission/tree/finance-stretch) | The finance stretch experiment, plus the signal-investigation finding described in the TL;DR above. |
| [`optimizer-extensions`](https://github.com/youngleox1/deeter-submission/tree/optimizer-extensions) | Exploring two more optimizer variants: LAMB, and a per-head Muon that orthogonalizes each attention head's weight sub-block independently. |

Git tags (`v0.1.0-scaffold` through `v0.5.0-nero-fix`) mark milestones in
`master`'s history — see `git tag -n` or the repo's Tags page.

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

Scoped to four on this branch — LAMB and a per-head Muon variant are explored
on [`optimizer-extensions`](https://github.com/youngleox1/deeter-submission/tree/optimizer-extensions)
instead of expanding this branch's grid; a full Modula/modular-norm
implementation was scoped out entirely for time-budget reasons.

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

<details id="basin-width-computation">
<summary><b>How basin width is actually computed</b> (click to expand — the exact formula and two caveats it implies)</summary>

(`analysis.ipynb`, not just described in prose): for a given optimizer, take
its own 9-point log-spaced LR grid, keep the subset of grid points whose
mean-across-seeds best validation loss is `<= threshold` (where
`threshold = AdamW's best loss x (1 + X)` — the same absolute threshold for
every optimizer, anchored to AdamW), and report

```
log10_basin_width = log10(max LR in that subset) - log10(min LR in that subset)
```

Two things worth being explicit about rather than leaving implicit:

- This is a **grid-resolution-limited estimate**, not the true continuous
  basin. The actual threshold-crossing point lies somewhere between two
  adjacent grid points; since the grid is evenly log-spaced at 0.375
  decades/step, the reported width is always a multiple of 0.375 and is a
  systematic *underestimate* of the true width (it can't resolve anything
  finer than one grid step).
- The code takes `max - min` of **all** qualifying grid points, without
  checking they're contiguous. If a non-qualifying point were sandwiched
  between two qualifying ones (a "hole" in the basin), this formula would
  silently report the full outer span rather than flagging the gap. In
  this project's actual data every qualifying set happens to be a
  contiguous run — checked with a programmatic assertion in
  `analysis.ipynb`, not just eyeballed — but that's a property of these
  particular loss curves being roughly unimodal in LR, not a guarantee the
  method provides.

</details>

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

## Data

- **Core:** char-level [tiny-Shakespeare](https://github.com/karpathy/char-rnn)
  corpus (public domain, ~1.1MB), vendored directly in this repo
  (`src/data/tinyshakespeare.txt`) rather than fetched at runtime, so the
  core experiment doesn't depend on network access to reproduce. See
  `src/data/text.py`.
- **Finance:** public daily OHLCV data pulled via the `yfinance` package for a
  small, fixed list of liquid US equity tickers, over a fixed historical
  window (not "most recent N days," so results don't shift if re-run
  later). This is external, freely available market data; no proprietary or
  non-public data is used. Cached CSVs are vendored in the repo for the same
  offline-reproducibility reason as the text corpus. See `src/data/finance.py`
  for the exact fetch logic and ticker list.

## Repo layout

```
src/
  model.py            small decoder-only transformer
  optimizers.py       AdamW/SGD baselines + Nero + Muon (from-scratch, checked against references)
  train.py            single training run
  sweep.py            LR x optimizer x seed grid driver
  eval_finance.py     directional-accuracy / Brier-score eval (finance stretch)
  data/
    text.py           core experiment data loader (vendored corpus)
    finance.py        finance stretch data loader (vendored cache + yfinance)
configs/              core_sweep.yaml, finance_sweep.yaml, LayerNorm-affine ablation config
results/              sweep outputs (csv) and generated plots
tests/                unit/smoke tests (see How to run)
analysis.ipynb        generates all core-experiment plots/tables from results/*.csv
scripts/run_all.sh    reproduce core experiment + ablation end to end
```

(Finance-specific files — `results/finance/`, `analysis_finance.ipynb`,
`scripts/investigate_finance_signal.py` — live on the `finance-stretch`
branch; `Lamb`/per-head `Muon` and their sweep config live on
`optimizer-extensions`. See [Branches](#branches).)

## How to run

```bash
pip install -r requirements.txt
pytest                          # run test suite
python -m src.sweep --config configs/core_sweep.yaml
python -m src.sweep --config configs/ablation_nero_no_ln_affine.yaml
# or, to do all of the above in one go:
bash scripts/run_all.sh
```

## Results

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

### Finance stretch: metrics and preprocessing, precisely

Three versions of this experiment follow below (v1/v2/v3). Rather than
re-explain the same metrics and preprocessing three times, here is the
full methodology once, with justification, so each version's results
section can just report numbers against it.

**Metrics** (`src/eval_finance.py` for the two directional-accuracy
baselines and Brier score; `src/train.py`/`src/sweep.py` for loss):

- **Validation cross-entropy loss**, compared against the uniform-
  distribution baseline `ln(n_bins)` (a model that has learned nothing
  gets exactly this value) — the same metric the core experiment uses,
  letting the basin-width/divergence-rate methodology carry over
  unchanged (see Success criteria above).
- **Persistence baseline** (originally the only baseline reported, now
  known to be too weak on its own — see below): predicts that tomorrow's
  direction repeats today's.
- **Majority-class baseline** (added after review; see the worked
  example below): predicts the single most common direction in the
  *training* period, unconditionally, every time.
- **Brier score**: mean squared error between the model's predicted
  P(up) and the realized 0/1 outcome. Reported alongside accuracy because
  a model can match accuracy while being badly overconfident or
  underconfident — Brier score is the more quant-relevant property for
  anything downstream that would use the probability, not just the
  argmax.

**Why two directional baselines, not one — a worked example, not just an
assertion.** This project initially reported only the persistence
baseline, and read "AdamW-family model beats persistence" as evidence
of a real edge. That was an incomplete comparison. Here is why, with the
actual numbers:

These six tickers have real positive drift — daily P(up) is
**51.8%-54.6%** per ticker over the full 2015-2025 history (SPY: 54.6%),
confirmed by direct computation on the cached price data, not assumed.
So "stocks went up more than they went down" is true. But persistence
accuracy under an i.i.d.-returns assumption is `p² + (1-p)²`, a function
that is **quadratically insensitive to drift near p=0.5**: even a real
5-point drift (p=0.55) only predicts ~50.4% persistence accuracy, nowhere
near 55%. On top of that, this project's own signal investigation
(`scripts/investigate_finance_signal.py`) already found real *negative*
lag-1 autocorrelation (short-term reversal) for 4 of 6 tickers — which
works directly against persistence, suppressing it further. Measured
directly on the actual time-ordered val split used throughout this
experiment:

| Baseline | Mean accuracy (val split) |
|---|---|
| Persistence | 52.0% |
| **Majority-class** | **56.3%** |

The majority-class baseline is not subject to either suppression effect
and is meaningfully stronger here. A model beating persistence but not
majority-class has not demonstrated an edge — it may simply have learned
to lean toward the majority class, which persistence itself fails to
capture. Both are reported from here on; a result is only claimed as a
real finding if it beats the **stronger** of the two.

**Data preprocessing, with justification for each choice:**

- **Quantile-bin discretization** (`ReturnTokenizer`, any `n_bins`): bin
  edges are fit on train data only (never val), so tokenization itself
  can't leak val-period statistics — a subtler leak than a bad train/val
  split but just as real, and specifically tested for (see `scripts` /
  `tests/test_finance_no_leakage.py`).
- **Volatility scaling** (`volatility_scale()`, v2/v3 only): divides each
  return by its trailing realized volatility (rolling `window=20`-day
  std, strictly causal via `.rolling(window).std().shift(1)` — the shift
  is what excludes the current day from its own scale estimate).
  Standard preprocessing for heavy-tailed, heteroscedastic returns
  ([volatility-derived features are commonly used inputs in recent
  applied work](https://pmc.ncbi.nlm.nih.gov/articles/PMC11577217/);
  GARCH-derived volatility as a neural-net input feature is an
  established pattern in the forecasting literature) — checked against
  current literature, not assumed.
- **Binary vs. 8-bin target** (v1 vs. v2/v3): binary classification of
  direction is standard practice for this exact task in recent
  (2024-2025) applied deep learning work on stock direction prediction,
  not fine-grained discretization ([Predicting daily stock price
  directions with deep learning
  models](https://www.sciencedirect.com/science/article/pii/S2666827025001276),
  [Stock Market Prediction Using ML/DL: A
  Review](https://www.mdpi.com/2673-9909/5/3/76)).
- **Continuous input** (v3 only, `continuous_input=True`): a real-valued
  linear projection instead of a token-embedding lookup for the input
  side, while the output head stays a discrete classifier — see the v3
  section below for why.
- **Time-ordered, per-ticker train/val split**, never a random shuffle of
  windows, and windows never cross a ticker boundary (unchanged across
  all three versions) — the standard walk-forward discipline for
  financial time series, avoiding the lookahead a random split would
  introduce.

### Finance stretch results (v1): 8-bin discretization

Same setup as the core experiment (4 optimizers x 9 LRs x 3 seeds = 108
runs, 500 steps, same code path via the data factory, rerun against the
corrected Nero). Raw results: `results/finance/sweep_results.csv`.
Analysis: `analysis_finance.ipynb`, figures in `results/finance/*.png`.
Directional-accuracy numbers below were recomputed against the corrected
dual-baseline eval (see previous section) — the original run only ever
needed retraining + re-evaluation, not a full sweep rerun.

**First finding: there isn't enough learnable signal here for the
basin-width comparison to mean anything, and that's reported as the
honest result rather than forced into a transfer/no-transfer verdict.**
With `n_bins=8`, a model that has learned nothing gets exactly
`ln(8) ≈ 2.079` (uniform-distribution cross-entropy). Every optimizer's
loss clusters within **1.5% of that baseline** across nearly the entire
LR range (see `results/finance/loss_vs_lr.png` — note the y-axis range
compared to the core experiment's version of the same plot).

| Optimizer | Best LR | Model directional accuracy | Persistence baseline | Majority baseline | Brier score |
|---|---|---|---|---|---|
| AdamW | 9.5e-5 | 49.3% | 50.8% | 53.3% | 0.251 |
| SGD | 0.30 | 50.2% | 50.8% | 53.3% | 0.251 |
| Nero | 0.071 | 49.1% | 50.8% | 53.3% | 0.251 |
| Muon | 6.3e-4 | 49.4% | 50.8% | 53.3% | 0.251 |

**None of the four optimizers beat either naive baseline** — all four sit
below even the weaker persistence baseline, let alone the stronger
majority-class one — and Brier scores (~0.25) are indistinguishable from
a coin flip (a predictor that always outputs 50/50 gets exactly 0.25 by
construction). Taken at face value, the basin-width finding from the
core experiment doesn't get a meaningful transfer test here — not because
it failed to transfer, but because there wasn't enough of a loss
landscape being fit for optimizer choice to matter.

**Second finding, and the more important one: that "no signal" reading
turned out to be about the pipeline, not the data.** Prompted by how
clean the null result above looked, a separate, much simpler check
(`scripts/investigate_finance_signal.py`,
`results/finance/signal_investigation.txt`) was run directly on the
cached price data, independent of the neural pipeline entirely:

- **Autocorrelation** at lag 1 is highly significant for SPY, AAPL, MSFT,
  and JPM (Ljung-Box p < 0.0001 for 3 of those 4), consistent with the
  well-documented short-term reversal effect in daily equity returns.
- **A trivial 5-lag logistic regression** (same time-ordered 85/15 split
  as this project's own loader) **beats the naive persistence baseline on
  every single one of the 6 tickers** — mean test accuracy 56.4% vs. 52.0%
  naive.

So there **is** real, modest, well-known signal in this exact data. The
honest conclusion is therefore not "no exploitable signal exists" (market
efficiency) — it's "**a simple linear method finds a real signal that
this specific neural pipeline fails to extract**." Most likely cause, not
confirmed: discretizing returns into 8 quantile bins and optimizing full
next-bin cross-entropy is a much noisier, higher-entropy objective than
directly predicting binary direction from 5 lagged values, and a weak
linear effect can easily get swamped in that harder task within only 500
steps. This is a genuine limitation of this project's finance pipeline,
not a claim about market efficiency — and it's a materially different,
more useful thing to have learned than the first finding alone would have
suggested.

An unexplored, clearly-labeled follow-up (not attempted here, due to
time): swap the 8-way bin cross-entropy objective for a direct binary
up/down classification head on the same transformer trunk, to test
whether that alone closes the gap to the simple logistic baseline.

### Finance stretch results (v2): binary direction + volatility scaling

Follow-up on the v1 follow-up above, motivated by two things: a literature
check on common practice for neural direction-prediction (not just
assumption — see citations below), and v1's own diagnosis that the 8-bin
objective was the likely culprit.

**What changed from v1, precisely:**

- **Objective: `n_bins` 8 → 2.** No new code needed — `ReturnTokenizer`
  already supports arbitrary bin counts via quantile binning, so `n_bins=2`
  is a plain binary up/down split at the median. Literature check: binary
  classification of direction is standard practice for this exact task in
  recent (2024-2025) applied deep learning work on stock direction
  prediction, not fine-grained discretization ([Predicting daily stock
  price directions with deep learning
  models](https://www.sciencedirect.com/science/article/pii/S2666827025001276),
  [Stock Market Prediction Using ML/DL: A
  Review](https://www.mdpi.com/2673-9909/5/3/76)).
- **Preprocessing: volatility scaling added** (`src/data/finance.py`'s new
  `volatility_scale()`). Each return is divided by its trailing realized
  volatility (rolling `window=20`-day std, strictly causal — computed via
  `.rolling(window).std().shift(1)`, where the shift is what excludes the
  current day from its own scale estimate) before tokenization. Standard
  preprocessing for heavy-tailed, heteroscedastic returns
  ([volatility-derived features are commonly used inputs in recent applied
  work](https://pmc.ncbi.nlm.nih.gov/articles/PMC11577217/); GARCH-derived
  volatility as a neural-net input feature is an established pattern in
  the forecasting literature).
- **Optimizer comparison narrowed to AdamW + Muon** (SGD and Nero dropped
  for this revision, per explicit request — not because v1 found them
  irrelevant).

**Setup, exactly** (`configs/finance_v2_sweep.yaml`,
`results/finance/v2_sweep_results.csv`):

| | Value |
|---|---|
| Model | same `DecoderOnlyTransformer`: `d_model=128, n_layers=4, n_heads=4, mlp_ratio=4, max_seq_len=64` |
| Data | `n_bins=2, vol_scale=True, vol_window=20, val_fraction=0.15`, same 6 tickers / 2015-2025 window as v1 |
| Training | `batch_size=64, seq_len=64, max_steps=500` (flat LR, no schedule — the `use_cosine_schedule` feature added on `optimizer-extensions` hasn't been merged into this branch) |
| Sweep | AdamW + Muon, same two 9-point LR grids as v1/`optimizer_extensions_sweep.yaml`, 3 seeds each = 54 runs, 0 diverged |
| Metrics | loss vs. `ln(2) ≈ 0.693` uniform baseline (replacing v1's `ln(8) ≈ 2.079`); directional accuracy vs. **both** naive baselines; Brier score — see the metrics section above |

**Results:**

| Optimizer | Best LR | Best val loss | Gap below uniform | Model dir. accuracy | Persistence baseline | Majority baseline | Brier score |
|---|---|---|---|---|---|---|---|
| AdamW | 0.040 | 0.6916 | 0.23% | 47.9% | 50.7% | 54.0% | 0.251 |
| Muon | 0.6325 | 0.6908 | 0.34% | 54.0% | 50.7% | 54.0% | 0.249 |

**Revision, not just an update: the "genuine edge" claim in an earlier
version of this section was wrong, and is retracted here rather than
quietly edited away.** That version only compared against persistence
(54.0% vs. 50.7%) and read Muon's result as a real, if modest, edge. Once
the majority-class baseline was added (prompted by a review question
about why the naive baseline sits so close to 50% despite real positive
drift — see the metrics section above), the picture changes: **Muon's
54.0% is not a beat, it is an exact tie with the majority-class
baseline's 54.0%.** That is not a coincidence — it is the signature of a
model that has learned to lean toward the majority class and nothing
more (AdamW's 47.9%, below both baselines, is consistent with a model
that hasn't even learned that much). Neither optimizer demonstrates a
real directional edge in v2.

**By loss, v2 also found *less* structure than v1** — the gap below the
uniform baseline shrank from v1's 1.3-1.4% to v2's 0.2-0.3%, consistent
with the corrected directional-accuracy picture rather than in tension
with it, as an earlier version of this section framed it.

**A likely confound, identified rather than glossed over:** switching to
`n_bins=2` didn't just simplify the *target* — because input and target
tokens come from the same tokenized stream in this next-token-prediction
setup, it also binarized the model's *input*. The model can now only see
whether each historical day was up or down, never the *magnitude* of the
move — exactly the information the simple 5-lag logistic-regression
baseline (which used raw continuous returns as features) still has full
access to. So v1-vs-v2 isn't a clean test of "does the objective alone
matter" — the input representation changed too, in a way that plausibly
works against the fix rather than for it. This is the most likely reason
v2 doesn't show a bigger, cleaner improvement.

### Finance stretch results (v3): continuous input via linear projection

Implements the v2 section's planned fix: feeds continuous (volatility-
scaled) returns as real-valued input via `nn.Linear(1, d_model)`
(`ModelConfig.continuous_input`), instead of a discrete token-embedding
lookup, while the output head is unchanged (still classifies into
`n_bins=2` discrete bins) — isolating whether it's the *objective*
(v1→v2) or the *input representation* (the v2 confound) that matters
more.

**Setup, exactly** (`configs/finance_v3_sweep.yaml`,
`results/finance/v3_sweep_results.csv`): identical to v2 —
`d_model=128, n_layers=4, n_heads=4, mlp_ratio=4, max_seq_len=64`,
`n_bins=2, vol_scale=True, vol_window=20`, AdamW + Muon, same two LR
grids, 3 seeds, 500 steps — with `continuous_input=True` set on both
`data` and `model`, and the target held at v2's binary classification.
54 runs, 0 diverged.

**Results:**

| Optimizer | Best LR | Best val loss | Gap below uniform | Model dir. accuracy | Persistence baseline | Majority baseline | Brier score |
|---|---|---|---|---|---|---|---|
| AdamW | 0.0949 | 0.6908 | 0.33% | 47.3% | 50.8% | 54.0% | 0.250 |
| Muon | 0.0474 | 0.6916 | 0.23% | 49.2% | 50.8% | 54.0% | 0.250 |

**The fix did not help — if anything, v3 is slightly worse than v2.**
Neither optimizer beats either baseline; both sit further below the
majority-class baseline than v2's AdamW did, and v2's Muon (which at
least tied the majority baseline) isn't matched by either v3 run. Loss
is essentially unchanged from v2 (0.2-0.3% below uniform either way).

**What this rules out, and what it doesn't.** The input-binarization
confound was a real, identified mechanism — the model genuinely can no
longer see move magnitude in discrete-input mode — but fixing it alone
does not close the gap to the simple 5-lag logistic-regression baseline
(56.4% mean test accuracy, from the v1 signal investigation). Plausible
remaining explanations, none confirmed: 500 steps and a single scalar
input per position may simply be too little signal/capacity for this
tiny transformer to find what a 5-feature linear model finds easily; the
binary cross-entropy objective, even with continuous input, may still be
a harder optimization target than the logistic regression's directly-
supervised setup; or the model needs the same longer-training-plus-
schedule treatment the core experiment's optimizer comparison got (see
`optimizer-extensions`' `use_cosine_schedule`, not yet merged into this
branch — see the note in Limitations).

<details id="limitations">
<summary><h2>Limitations (click to expand)</h2></summary>

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
  of a reference implementation. (Muon wasn't in any version of PyTorch
  when this was written — it was added as `torch.optim.Muon` in PyTorch
  2.9; this project's installed torch is 2.7.1, which doesn't have it.)
  Checked against the now-official implementation
  (github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py) while
  investigating a per-head variant on a separate branch: our Newton-Schulz
  coefficients and LR-adjustment formula match exactly, but our momentum
  is **plain (heavy-ball)**, not the native implementation's **default
  Nesterov** momentum (orthogonalization input is `g_t + momentum * B_t`,
  not just the momentum buffer `B_t`) — matches native Muon's
  `nesterov=False` option, just not its default. We also apply no weight
  decay (native defaults to 0.1, decoupled). Neither difference has been
  tested for impact on the results above; noted here rather than silently
  left as an unexamined assumption. **Nero was initially a much more
  meaningfully different reimplementation** (missing mean-centering,
  wrong projection target, sum-vs-mean second moment, spurious momentum
  on 1D params) until corrected against the actual reference code
  (github.com/jxbz/nero) — see `v0.5.0-nero-fix` tag. The corrected
  version's remaining underperformance vs. AdamW is treated as a genuine
  finding, not attributed to further unverified implementation
  differences.
- **LAMB and full Modula/modular-norm were scoped out** of this branch's
  optimizer comparison — LAMB and a per-head Muon variant are explored on
  `optimizer-extensions` instead (see [Branches](#branches)); Modula was
  scoped out entirely for time-budget reasons.
- Muon's per-parameter-group hybrid design (orthogonalized update for 2D
  hidden matrices, AdamW-style fallback for the rest) means its "LR" in the
  sweep only varies the Muon branch; the fallback-branch LR is held fixed
  (see `configs/core_sweep.yaml`). A full 2D sweep over both would be more
  thorough but was out of scope here.
- **The finance stretch's neural pipeline fails to extract signal that
  demonstrably exists** in this exact data (see Finance stretch results
  above) — likely because discretizing returns into 8 bins and optimizing
  full next-bin cross-entropy is a much noisier objective than directly
  predicting binary direction. Swapping in a direct binary classification
  head is the natural fix; not attempted here due to time.

</details>

<details id="references">
<summary><h2>References (click to expand)</h2></summary>

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
  above; since incorporated into PyTorch core as `torch.optim.Muon` in
  2.9, see Limitations.)
- You, Y. et al. "Large Batch Optimization for Deep Learning: Training BERT
  in 76 Minutes." ICLR 2020. (LAMB — implemented on `optimizer-extensions`,
  see Branches above.)
- "Predicting daily stock price directions with deep learning models."
  ScienceDirect, 2025. (Literature check behind the v2 finance stretch's
  switch to binary direction classification.)
- "Stock Market Prediction Using Machine Learning and Deep Learning
  Techniques: A Review." MDPI, 2024. (Same, plus confirms volatility/
  technical-indicator features as standard practice.)

</details>
