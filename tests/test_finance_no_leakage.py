import numpy as np
import pandas as pd
import pytest
import torch

from src.data import finance as finance_module
from src.data.finance import FinanceReturns, ReturnTokenizer, volatility_scale


def _make_fake_close_series(n=300, seed=0, val_outlier=False, val_fraction=0.2):
    rng = np.random.RandomState(seed)
    log_rets = rng.normal(0, 0.01, size=n)
    if val_outlier:
        split = int(n * (1 - val_fraction))
        log_rets[split:] += rng.choice([-1, 1], size=n - split) * 5.0  # huge spike, VAL-only
    price = 100 * np.exp(np.cumsum(log_rets))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(price, index=dates, name="Close")


def _patch_fetch(monkeypatch, series_by_ticker):
    def _fetch(ticker, start, end, cache_dir):
        return series_by_ticker[ticker]
    monkeypatch.setattr(finance_module, "_fetch_ticker_close", _fetch)


def test_tokenizer_bin_edges_unaffected_by_extreme_val_period_values(monkeypatch):
    """Regression test for the specific leakage failure mode this loader
    is designed to avoid: if the tokenizer were (incorrectly) fit on the
    full series, an extreme spike placed only in the val period would
    shift the bin edges. Since the spike occurs strictly after the split,
    the train-period prices/returns are byte-for-byte identical between
    the two datasets below -- so fit-on-train-only means the edges must
    be exactly equal despite the val period being wildly different.
    """
    _patch_fetch(monkeypatch, {"FAKE1": _make_fake_close_series(seed=1, val_outlier=False)})
    ds_clean = FinanceReturns(tickers=["FAKE1"], val_fraction=0.2, n_bins=8)

    _patch_fetch(monkeypatch, {"FAKE1": _make_fake_close_series(seed=1, val_outlier=True)})
    ds_spiked = FinanceReturns(tickers=["FAKE1"], val_fraction=0.2, n_bins=8)

    assert np.allclose(ds_clean.tokenizer.bin_edges, ds_spiked.tokenizer.bin_edges), (
        "bin edges changed when only the VAL period was perturbed -- "
        "the tokenizer must be fit on train data only"
    )


def test_split_is_time_ordered_and_lengths_match_val_fraction(monkeypatch):
    _patch_fetch(monkeypatch, {"FAKE1": _make_fake_close_series(n=300, seed=2)})
    ds = FinanceReturns(tickers=["FAKE1"], val_fraction=0.2, n_bins=8)

    total_len = len(ds.train_streams[0]) + len(ds.val_streams[0])
    assert total_len == 299  # 300 prices -> 299 log returns
    assert abs(len(ds.val_streams[0]) / total_len - 0.2) < 0.01


def test_get_batch_shapes(monkeypatch):
    _patch_fetch(monkeypatch, {
        "FAKE1": _make_fake_close_series(n=300, seed=3),
        "FAKE2": _make_fake_close_series(n=300, seed=4),
    })
    ds = FinanceReturns(tickers=["FAKE1", "FAKE2"], val_fraction=0.2, n_bins=8)
    x, y = ds.get_batch("train", batch_size=6, seq_len=20)
    assert x.shape == (6, 20)
    assert y.shape == (6, 20)
    assert x.dtype == y.dtype


def test_get_batch_raises_when_no_stream_long_enough(monkeypatch):
    _patch_fetch(monkeypatch, {"FAKE1": _make_fake_close_series(n=50, seed=5)})
    ds = FinanceReturns(tickers=["FAKE1"], val_fraction=0.5, n_bins=8)
    with pytest.raises(ValueError):
        ds.get_batch("val", batch_size=2, seq_len=1000)


def test_return_tokenizer_bin_to_direction_sign_matches_bin_order():
    tokenizer = ReturnTokenizer(n_bins=4)
    train_returns = np.array([-0.05, -0.04, -0.01, -0.005, 0.005, 0.01, 0.04, 0.05])
    tokenizer.fit(train_returns)

    directions = tokenizer.bin_to_direction(np.arange(4))
    # lowest-index bins should be non-positive, highest-index bins non-negative
    assert directions[0] <= 0
    assert directions[-1] >= 0
    assert list(directions) == sorted(directions)


def test_volatility_scale_drops_exactly_window_days_from_the_start():
    rng = np.random.RandomState(0)
    returns = rng.normal(0, 0.01, size=100)
    scaled = volatility_scale(returns, window=20)
    assert len(scaled) == 100 - 20


def test_volatility_scale_matches_hand_computed_value():
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.05, -0.02])
    window = 3
    scaled = volatility_scale(returns, window=window)
    # first scaled value corresponds to day index 3 (0-indexed): trailing
    # window is returns[0:3] = [0.01, 0.02, -0.01], excluding day 3 itself.
    # Use pandas' std (ddof=1, sample std) to match the implementation --
    # numpy's default ddof=0 (population std) gives a different value.
    expected_std = pd.Series(returns[0:3]).std()
    expected_first = returns[3] / expected_std
    assert abs(scaled[0] - expected_first) < 1e-9


def test_volatility_scale_is_strictly_causal_no_lookahead():
    """Changing a return far in the future must not change the scaled
    value at an earlier day -- the defining no-lookahead property, same
    discipline as the tokenizer's train/val split test above.
    """
    rng = np.random.RandomState(0)
    returns = rng.normal(0, 0.01, size=100)
    window = 20

    scaled_original = volatility_scale(returns, window=window)

    returns_modified = returns.copy()
    returns_modified[90] = 10.0  # huge change, far in the future
    scaled_modified = volatility_scale(returns_modified, window=window)

    # scaled values for days well before index 90 must be identical
    assert np.allclose(scaled_original[:50], scaled_modified[:50])


def test_finance_returns_with_vol_scale_runs_end_to_end(monkeypatch):
    _patch_fetch(monkeypatch, {"FAKE1": _make_fake_close_series(n=300, seed=6)})
    ds = FinanceReturns(tickers=["FAKE1"], val_fraction=0.2, n_bins=2,
                         vol_scale=True, vol_window=20)
    x, y = ds.get_batch("train", batch_size=4, seq_len=10)
    assert x.shape == (4, 10)
    assert ds.vocab_size == 2


def test_continuous_input_x_is_float_and_matches_tokenizer_input(monkeypatch):
    """The defining correctness property of continuous_input mode: x must
    be the real continuous return values, aligned to the exact same
    (ticker, day) positions y's discrete bins come from. Verified by
    comparing against an identically-seeded discrete-mode loader: since
    both draw the same random (ticker, start) pairs, re-tokenizing
    continuous x must reproduce discrete x exactly, and y must be
    unaffected by continuous_input either way.
    """
    _patch_fetch(monkeypatch, {
        "FAKE1": _make_fake_close_series(n=300, seed=7),
        "FAKE2": _make_fake_close_series(n=300, seed=8),
    })
    ds = FinanceReturns(tickers=["FAKE1", "FAKE2"], val_fraction=0.2, n_bins=4,
                         continuous_input=True, seed=0)
    x, y = ds.get_batch("train", batch_size=8, seq_len=15)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert not torch.allclose(x, x.round())  # genuinely continuous, not integer-valued

    ds_discrete = FinanceReturns(tickers=["FAKE1", "FAKE2"], val_fraction=0.2, n_bins=4,
                                  continuous_input=False, seed=0)
    x_discrete, y_discrete = ds_discrete.get_batch("train", batch_size=8, seq_len=15)

    retokenized_x = torch.tensor(ds.tokenizer.transform(x.numpy().ravel())).view(x.shape)
    assert torch.equal(retokenized_x, x_discrete)
    assert torch.equal(y, y_discrete)  # y is unaffected by continuous_input, as designed


def test_majority_direction_train_matches_hand_computed_sign_distribution(monkeypatch):
    """majority_direction_train must reflect the actual majority sign of
    TRAIN raw returns -- not something derived from the tokenizer's bins.
    A moderate positive skew (57% up days, realistic in magnitude, unlike
    real market data's ~52-55%) still leaves the median near zero, so
    quantile bins still straddle it -- confirming majority_direction_train
    tracks the true marginal even when it's NOT recoverable from bin
    structure alone (bin counts are forced to exactly 50/50 by
    construction regardless of the marginal).
    """
    rng = np.random.RandomState(0)
    n = 200
    log_rets = rng.normal(0, 0.01, size=n)
    up_idx = rng.choice(n, size=int(n * 0.57), replace=False)
    log_rets[up_idx] = abs(log_rets[up_idx]) + 1e-5  # force these up, keep magnitudes modest

    price = 100 * np.exp(np.cumsum(log_rets))
    dates = pd.date_range("2020-01-01", periods=n + 1, freq="B")
    close = pd.Series(np.concatenate([[100.0], price]), index=dates, name="Close")

    _patch_fetch(monkeypatch, {"FAKE1": close})
    ds = FinanceReturns(tickers=["FAKE1"], val_fraction=0.2, n_bins=2)

    assert ds.majority_direction_train == 1.0
    # both quantile bins are ~50% of train BY CONSTRUCTION regardless of
    # the 57/43 true marginal -- majority_direction_train is not derivable
    # from this bin-count structure, which is exactly the point.
    assert (ds.tokenizer.bin_mean_return > 0).sum() == 1
    assert (ds.tokenizer.bin_mean_return < 0).sum() == 1
