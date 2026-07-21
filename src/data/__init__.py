"""Dataset factory so sweep.py / train.py's CLI are agnostic to which
data domain they're driving -- the core (text) and finance stretch
experiments reuse the exact same training/sweep code, only the `data`
config section differs.
"""
from typing import Any, Dict

from src.data.finance import FinanceReturns
from src.data.text import TinyShakespeare


def build_dataset(data_cfg: Dict[str, Any], seed: int):
    data_cfg = dict(data_cfg)
    data_type = data_cfg.pop("type")
    if data_type == "text":
        return TinyShakespeare(seed=seed, **data_cfg)
    if data_type == "finance":
        return FinanceReturns(seed=seed, **data_cfg)
    raise ValueError(f"unknown data type: {data_type!r}")
