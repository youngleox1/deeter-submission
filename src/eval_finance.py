"""Directional-accuracy and calibration evaluation for the finance stretch
experiment. Separate from train.py's cross-entropy loss (which drives the
basin-width/divergence-rate metrics shared with the core experiment) --
this answers a different, finance-specific question: is the model's
predicted direction any better than a naive baseline, and is its
confidence calibrated?

TWO naive baselines are reported, not one -- an earlier version of this
project only tracked persistence (predict yesterday's direction repeats)
and reported a model beating it as "a genuine edge." That was wrong: this
data has real positive drift (P(up) ~52-55% per ticker), but persistence
accuracy under an i.i.d. assumption is p^2+(1-p)^2, which is quadratically
INSENSITIVE to drift near p=0.5 (a 5-point drift barely moves it off 50%)
-- and real negative lag-1 autocorrelation (short-term reversal, found in
this project's own signal investigation) pulls it down further. A
majority-class baseline ("always predict train's more common direction")
is not similarly suppressed and is meaningfully stronger on this data
(~56% vs. persistence's ~52% on the actual val split) -- a result must
beat the STRONGER of the two to mean anything, not just persistence.

"Direction" for a bin is defined via that bin's TRAIN mean-return sign
(ReturnTokenizer.bin_to_direction), consistent with how the model was
trained -- not a separate re-derivation from raw prices.
"""
from typing import Any, Dict

import numpy as np
import torch


@torch.no_grad()
def evaluate_directional_metrics(
    model: torch.nn.Module, data, batch_size: int, seq_len: int,
    n_eval_batches: int, device: str = "cpu",
) -> Dict[str, Any]:
    model.eval()
    tokenizer = data.tokenizer
    direction_lookup = torch.tensor(
        np.sign(tokenizer.bin_mean_return), device=device, dtype=torch.float32
    )
    positive_mask = direction_lookup > 0
    continuous_input = getattr(data, "continuous_input", False)
    majority_direction = data.majority_direction_train

    n_correct_model = 0
    n_correct_persistence = 0
    n_correct_majority = 0
    n_total = 0
    brier_sum = 0.0

    for _ in range(n_eval_batches):
        x, y = data.get_batch("val", batch_size, seq_len, device)
        logits, _ = model(x, y)
        probs = torch.softmax(logits, dim=-1)
        pred_bin = probs.argmax(dim=-1)

        pred_dir = direction_lookup[pred_bin]
        actual_dir = direction_lookup[y]
        if continuous_input:
            # x is a real-valued return, not a bin id -- its own sign IS
            # the naive "yesterday's direction repeats" prediction, no
            # lookup table needed (and none is valid: direction_lookup is
            # indexed by discrete bin id, x here is a float).
            persistence_dir = torch.sign(x)
        else:
            persistence_dir = direction_lookup[x]  # predict yesterday's direction repeats

        n_correct_model += (pred_dir == actual_dir).sum().item()
        n_correct_persistence += (persistence_dir == actual_dir).sum().item()
        n_correct_majority += (actual_dir == majority_direction).sum().item()
        n_total += actual_dir.numel()

        p_up = probs[..., positive_mask].sum(dim=-1)
        outcome_up = (actual_dir > 0).float()
        brier_sum += ((p_up - outcome_up) ** 2).sum().item()

    return {
        "model_directional_accuracy": n_correct_model / n_total,
        "persistence_directional_accuracy": n_correct_persistence / n_total,
        "majority_directional_accuracy": n_correct_majority / n_total,
        "brier_score": brier_sum / n_total,
        "n_predictions": n_total,
    }
