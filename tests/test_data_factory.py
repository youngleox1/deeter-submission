import numpy as np
import pandas as pd
import pytest

from src.data import build_dataset
from src.data import finance as finance_module
from src.data.finance import FinanceReturns
from src.data.text import TinyShakespeare


def test_build_dataset_text():
    ds = build_dataset({"type": "text"}, seed=0)
    assert isinstance(ds, TinyShakespeare)


def test_build_dataset_finance(monkeypatch):
    def _fake_fetch(ticker, start, end, cache_dir):
        rng = np.random.RandomState(0)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=300)))
        return pd.Series(price, index=pd.date_range("2020-01-01", periods=300, freq="B"))

    monkeypatch.setattr(finance_module, "_fetch_ticker_close", _fake_fetch)
    ds = build_dataset({"type": "finance", "tickers": ["FAKE"], "n_bins": 4}, seed=0)
    assert isinstance(ds, FinanceReturns)


def test_build_dataset_unknown_type_raises():
    with pytest.raises(ValueError):
        build_dataset({"type": "not_a_real_type"}, seed=0)
