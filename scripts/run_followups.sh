#!/usr/bin/env bash
# Reproduce everything past the original core sweep + ablation: the
# schedule ablation, the longer-training check, the hero run (both fixes
# combined, full grid), and the analysis/plotting scripts that generate
# the README's headline figure, SEM-annotated tables, raw training
# curves, and the optimizer compute/memory measurements.
#
# Not run by scripts/run_all.sh -- this is substantially longer (roughly
# 3 hours on a single GPU, dominated by the hero sweep's 108 runs x 3000
# steps) and is kept separate so a quick sanity-check of the original
# result doesn't require reproducing all of it.
set -euo pipefail

export PYTHONPATH=.

python -m src.sweep --config configs/core_sweep_schedule_ablation.yaml
python -m src.sweep --config configs/core_longer_training_check.yaml
python -m src.sweep --config configs/core_sweep_hero.yaml

python scripts/analyze_schedule_and_longer_training.py
python scripts/analyze_with_sem.py
python scripts/plot_summary_aligned_lr.py
python scripts/capture_raw_curves.py
python scripts/measure_optimizer_cost.py
