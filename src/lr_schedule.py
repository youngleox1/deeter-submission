"""Linear-warmup, cosine-decay LR schedule.

Addresses two limitations flagged early in this project and left
unaddressed until now: no LR schedule was used anywhere (flat LR for the
whole run), and 500 steps is short relative to typical conventions for
this toy setup. Returns a MULTIPLIER in [min_lr_ratio, 1.0], not an
absolute LR, so it can scale any optimizer's base LR(s) uniformly.
"""
import math


def cosine_warmup_multiplier(step: int, total_steps: int, warmup_steps: int,
                              min_lr_ratio: float = 0.1) -> float:
    """step is 1-indexed (matches train.py's training loop). Linear ramp
    from 0 to 1 over the first warmup_steps, then cosine decay from 1
    down to min_lr_ratio over the remaining steps.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    if warmup_steps >= total_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine
