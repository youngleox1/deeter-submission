"""Finance stretch data loader.

Daily log returns are discretized into `n_bins` quantile bins and the
resulting bin sequence is treated as tokens -- so the exact same
DecoderOnlyTransformer / optimizer / train.py code used for the core text
experiment applies completely unchanged; only the tokenizer and data
source change. This is what keeps the "does the finding transfer to a new
domain" claim isolated to the domain, not conflated with an architecture
change (see README).

Data: public daily OHLCV via the `yfinance` package for a small, fixed
list of liquid US equity tickers, over a fixed historical date range
(not "most recent N days") so results are reproducible regardless of
when this is run. This is external, freely available market data --
disclosed explicitly here, no proprietary or non-public data is used.
Fetched data is cached to src/data/finance_cache/*.csv and vendored in
this repo (same reasoning as the vendored text corpus: reproducibility
without a live network dependency), fetched 2026-07-21.

Optional `vol_scale` preprocessing (see `volatility_scale()` below):
divides returns by a trailing realized-volatility estimate before
tokenization. Standard practice for taming heavy-tailed, heteroscedastic
returns (checked against current literature, not assumed) -- added
alongside a switch to n_bins=2 (binary up/down) as a direct test of the
project's own earlier diagnosis: an 8-bin discretization + full
next-bin cross-entropy is a much noisier objective than directly
predicting the binary quantity actually of interest, plausibly swamping
the weak signal a simple 5-lag logistic regression finds easily (see
scripts/investigate_finance_signal.py).
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-01"
_CACHE_DIR = Path(__file__).parent / "finance_cache"


class ReturnTokenizer:
    """Discretizes daily log returns into `n_bins` quantile bins. Bin
    edges (and each bin's mean-return sign, used later for directional
    accuracy) are fit on TRAIN data only -- fitting on the full series
    would leak val-period return statistics into the tokenization itself,
    which is a subtler leak than a bad model-level train/val split but
    just as real.
    """

    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_mean_return: Optional[np.ndarray] = None

    def fit(self, train_returns: np.ndarray) -> "ReturnTokenizer":
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bin_edges = np.quantile(train_returns, quantiles)
        bin_ids = np.digitize(train_returns, self.bin_edges)
        self.bin_mean_return = np.array([
            train_returns[bin_ids == b].mean() if np.any(bin_ids == b) else 0.0
            for b in range(self.n_bins)
        ])
        return self

    def transform(self, returns: np.ndarray) -> np.ndarray:
        assert self.bin_edges is not None, "call fit() before transform()"
        return np.digitize(returns, self.bin_edges)

    def bin_to_direction(self, bin_ids: np.ndarray) -> np.ndarray:
        """Map bin ids to {-1, 0, +1} using each bin's TRAIN mean-return
        sign -- for the directional-accuracy metric, not used in training.
        """
        assert self.bin_mean_return is not None, "call fit() before bin_to_direction()"
        return np.sign(self.bin_mean_return)[bin_ids]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


def volatility_scale(returns: np.ndarray, window: int = 20, eps: float = 1e-8) -> np.ndarray:
    """Divide each return by its trailing realized volatility (std of the
    PRECEDING `window` returns, strictly excluding the current day) --
    standard preprocessing for heavy-tailed, heteroscedastic returns
    before modeling (see README for the literature check behind this).

    Strictly causal: day t's scale uses only [t-window, t), never t
    itself -- the same no-lookahead discipline as the tokenizer's
    train/val split. Implemented via `.rolling(window).std().shift(1)`:
    the shift is what excludes the current day. The first `window` days
    (insufficient trailing history) are dropped, not estimated from a
    short/expanding window.
    """
    s = pd.Series(returns)
    trailing_std = s.rolling(window=window).std().shift(1)
    scaled = (s / trailing_std.clip(lower=eps)).values
    return scaled[~np.isnan(scaled)]


def _fetch_ticker_close(ticker: str, start: str, end: str, cache_dir: Path) -> pd.Series:
    cache_path = cache_dir / f"{ticker}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)["Close"]

    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} in [{start}, {end}]")

    cache_dir.mkdir(parents=True, exist_ok=True)
    df["Close"].to_csv(cache_path)
    return df["Close"]


class FinanceReturns:
    """Serves discretized daily-return-bin sequences from a fixed list of
    tickers. Split is time-ordered PER TICKER (train = earlier dates, val
    = later dates for that same ticker) -- never a random shuffle of
    windows, and windows never cross a ticker boundary.
    """

    def __init__(self, tickers: Optional[List[str]] = None, start: str = DEFAULT_START,
                 end: str = DEFAULT_END, n_bins: int = 8, val_fraction: float = 0.15,
                 seed: int = 0, cache_dir: Path = _CACHE_DIR,
                 vol_scale: bool = False, vol_window: int = 20,
                 continuous_input: bool = False):
        self.tickers = tickers or DEFAULT_TICKERS
        self.vol_scale = vol_scale
        self.continuous_input = continuous_input

        raw_returns = []
        for ticker in self.tickers:
            close = _fetch_ticker_close(ticker, start, end, cache_dir)
            log_ret = np.log(close).diff().dropna().values
            if vol_scale:
                log_ret = volatility_scale(log_ret, window=vol_window)
            raw_returns.append(log_ret)

        split_points = [int(len(r) * (1 - val_fraction)) for r in raw_returns]
        train_returns_concat = np.concatenate(
            [r[:s] for r, s in zip(raw_returns, split_points)]
        )
        self.tokenizer = ReturnTokenizer(n_bins=n_bins).fit(train_returns_concat)
        self.vocab_size = self.tokenizer.vocab_size

        # Majority-class direction, fit on TRAIN raw returns directly (not
        # via the tokenizer): quantile binning forces an exact 50/50 split
        # by construction at n_bins=2, which discards the marginal P(up)
        # information a majority-vote baseline needs -- computed here from
        # the actual sign distribution instead.
        self.majority_direction_train = (
            1.0 if (train_returns_concat > 0).mean() > 0.5 else -1.0
        )

        self.train_streams = [
            torch.tensor(self.tokenizer.transform(r[:s]), dtype=torch.long)
            for r, s in zip(raw_returns, split_points) if s > 0
        ]
        self.val_streams = [
            torch.tensor(self.tokenizer.transform(r[s:]), dtype=torch.long)
            for r, s in zip(raw_returns, split_points) if len(r) - s > 0
        ]
        # continuous (untokenized) streams, built from the exact same per-
        # ticker slices/filter as the discrete streams above, so index i in
        # train_continuous_streams always corresponds to the exact same
        # ticker/days as index i in train_streams (needed so x and y stay
        # aligned when continuous_input=True pulls x from one and y from
        # the other).
        self.train_continuous_streams = [
            torch.tensor(r[:s], dtype=torch.float32)
            for r, s in zip(raw_returns, split_points) if s > 0
        ]
        self.val_continuous_streams = [
            torch.tensor(r[s:], dtype=torch.float32)
            for r, s in zip(raw_returns, split_points) if len(r) - s > 0
        ]
        self._generator = torch.Generator().manual_seed(seed)

    def get_batch(self, split: str, batch_size: int, seq_len: int, device="cpu"):
        discrete_streams = self.train_streams if split == "train" else self.val_streams
        continuous_streams = (self.train_continuous_streams if split == "train"
                               else self.val_continuous_streams)
        eligible_idx = [i for i, s in enumerate(discrete_streams) if len(s) > seq_len + 1]
        if not eligible_idx:
            raise ValueError(f"no '{split}' ticker stream long enough for seq_len={seq_len}")

        xs, ys = [], []
        for _ in range(batch_size):
            stream_idx = eligible_idx[
                torch.randint(0, len(eligible_idx), (1,), generator=self._generator).item()
            ]
            discrete_stream = discrete_streams[stream_idx]
            max_start = len(discrete_stream) - seq_len - 1
            start = torch.randint(0, max_start, (1,), generator=self._generator).item()

            if self.continuous_input:
                xs.append(continuous_streams[stream_idx][start: start + seq_len])
            else:
                xs.append(discrete_stream[start: start + seq_len])
            ys.append(discrete_stream[start + 1: start + seq_len + 1])

        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        return x, y
