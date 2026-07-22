"""Directional-accuracy and calibration evaluation for the finance stretch
experiment. Separate from train.py's cross-entropy loss (which drives the
basin-width/divergence-rate metrics shared with the core experiment) --
this answers a different, finance-specific question: is the model's
predicted direction any better than a naive persistence baseline, and is
its confidence calibrated?

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

    n_correct_model = 0
    n_correct_naive = 0
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
            naive_dir = torch.sign(x)
        else:
            naive_dir = direction_lookup[x]  # persistence: predict yesterday's direction repeats

        n_correct_model += (pred_dir == actual_dir).sum().item()
        n_correct_naive += (naive_dir == actual_dir).sum().item()
        n_total += actual_dir.numel()

        p_up = probs[..., positive_mask].sum(dim=-1)
        outcome_up = (actual_dir > 0).float()
        brier_sum += ((p_up - outcome_up) ** 2).sum().item()

    return {
        "model_directional_accuracy": n_correct_model / n_total,
        "naive_directional_accuracy": n_correct_naive / n_total,
        "brier_score": brier_sum / n_total,
        "n_predictions": n_total,
    }
