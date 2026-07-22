#!/usr/bin/env bash
# Reproduce the core experiment + LayerNorm-affine ablation end to end.
set -euo pipefail

pip install -r requirements.txt
pytest

python -m src.sweep --config configs/core_sweep.yaml
python -m src.sweep --config configs/ablation_nero_no_ln_affine.yaml
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
