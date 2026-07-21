"""Independent check: is there ANY detectable signal in this exact daily
return dataset, using simple/well-understood methods, regardless of our
neural model's capacity, tokenization, or optimizer choice?

Run after the finance stretch sweep came back near-uniform-baseline with
no directional edge over a naive persistence baseline (see README) --
this checks whether that null result is about the data (no signal to
find) or about the pipeline (signal exists, our approach doesn't find
it). Uses the same cached price data and the same time-ordered 85/15
split logic as src/data/finance.py, but a completely independent,
much simpler modeling approach (autocorrelation + logistic regression
on raw lagged returns, no discretization).

Output captured in results/finance/signal_investigation.txt.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LogisticRegression

TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
CACHE_DIR = Path(__file__).parent.parent / "src" / "data" / "finance_cache"


def main():
    all_returns = {}
    for t in TICKERS:
        close = pd.read_csv(CACHE_DIR / f"{t}.csv", index_col=0, parse_dates=True)["Close"]
        all_returns[t] = np.log(close).diff().dropna().values

    print("=" * 70)
    print("1. Autocorrelation of daily log returns, lags 1-5")
    print("=" * 70)
    for t, r in all_returns.items():
        n = len(r)
        sig_band = 1.96 / np.sqrt(n)
        acfs = [np.corrcoef(r[:-lag], r[lag:])[0, 1] for lag in range(1, 6)]
        flags = ["*" if abs(a) > sig_band else " " for a in acfs]
        acf_str = "  ".join(f"{a:+.4f}{f}" for a, f in zip(acfs, flags))
        print(f"{t:6s} (n={n:4d}, 95% band=+/-{sig_band:.4f}): {acf_str}")

    print()
    print("Pooled (all tickers concatenated):")
    pooled = np.concatenate(list(all_returns.values()))
    n = len(pooled)
    sig_band = 1.96 / np.sqrt(n)
    acfs = [np.corrcoef(pooled[:-lag], pooled[lag:])[0, 1] for lag in range(1, 6)]
    flags = ["*" if abs(a) > sig_band else " " for a in acfs]
    print("  ".join(f"{a:+.4f}{f}" for a, f in zip(acfs, flags)), f" (95% band=+/-{sig_band:.4f})")

    print()
    print("=" * 70)
    print("2. Ljung-Box test (joint significance of lags 1-5 autocorrelation)")
    print("=" * 70)
    for t, r in all_returns.items():
        lb = acorr_ljungbox(r, lags=[5], return_df=True)
        print(f"{t:6s}: LB stat={lb['lb_stat'].iloc[0]:.3f}  p-value={lb['lb_pvalue'].iloc[0]:.4f}")
    lb_pooled = acorr_ljungbox(pooled, lags=[5], return_df=True)
    print(f"pooled: LB stat={lb_pooled['lb_stat'].iloc[0]:.3f}  p-value={lb_pooled['lb_pvalue'].iloc[0]:.4f}")

    print()
    print("=" * 70)
    print("3. Simple logistic regression: predict sign(return_t) from lags 1-5")
    print("   Time-ordered 85/15 split per ticker (same split logic as FinanceReturns)")
    print("=" * 70)
    n_lags = 5
    train_acc_all, test_acc_all, naive_acc_all = [], [], []
    for t, r in all_returns.items():
        X, y = [], []
        for i in range(n_lags, len(r)):
            X.append(r[i - n_lags:i])
            y.append(1 if r[i] > 0 else 0)
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        naive_pred = (X_test[:, -1] > 0).astype(int)
        naive_acc = (naive_pred == y_test).mean()

        print(f"{t:6s}: train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  naive_test_acc={naive_acc:.4f}  "
              f"test_beats_naive={'YES' if test_acc > naive_acc else 'no'}")
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        naive_acc_all.append(naive_acc)

    print()
    print(f"Mean across tickers: train={np.mean(train_acc_all):.4f}  test={np.mean(test_acc_all):.4f}  "
          f"naive={np.mean(naive_acc_all):.4f}")


if __name__ == "__main__":
    main()
