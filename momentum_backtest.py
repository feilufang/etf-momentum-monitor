#!/usr/bin/env python3
"""
Basic momentum backtest on the least-correlated ETF universes.

Signal   : Past N-day close-to-close return (multiple lookbacks tested).
Portfolio: Long top quartile, short bottom quartile. Equal weight.
           Dollar-neutral: long $1, short $1 gross per side.
Rebal    : Every REBAL_DAYS trading days (default 5 = weekly).

Inputs
------
    results_corr/selected_K*_*.csv   from select_uncorrelated_etfs.py
    data/daily/all.parquet

Outputs (in results_momentum/)
-------------------------------
    stats_summary.csv    Sharpe / Ann.Ret / MaxDD for every combo
    cumret_<combo>.png   cumulative return charts
    positions_<combo>.csv  daily position log (optional --save-positions)

Usage
-----
    python momentum_backtest.py
    python momentum_backtest.py --lookbacks 21 63 126
    python momentum_backtest.py --rebal 21 --plot
"""

import argparse
import sys
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DATA_DIR   = _HERE.parent.parent / "data"
DAILY_FILE = DATA_DIR / "daily/all.parquet"
CORR_DIR   = _HERE / "results_corr"

REBAL_DAYS    = 5        # weekly
DEFAULT_LOOKBACKS = [21, 63, 126, 252]   # 1M, 3M, 6M, 12M


# ── Load universe lists ────────────────────────────────────────────────────────

def load_universes() -> dict[str, list[str]]:
    universes = {}
    for path in sorted(CORR_DIR.glob("selected_K*_*.csv")):
        df = pd.read_csv(path)
        label = path.stem.replace("selected_", "")   # e.g. "K20_hierarchical"
        universes[label] = df["ticker"].tolist()
    if not universes:
        raise FileNotFoundError(
            f"No selected_K*.csv files found in {CORR_DIR}. "
            "Run select_uncorrelated_etfs.py first."
        )
    return universes


# ── Load price data ────────────────────────────────────────────────────────────

def load_prices(tickers: set[str]) -> pd.DataFrame:
    daily = pd.read_parquet(
        DAILY_FILE, columns=["date", "ticker", "close"]
    )
    daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
    close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    close.sort_index(inplace=True)
    close = close.ffill()
    return close


# ── Backtest engine ────────────────────────────────────────────────────────────

def _erc_weights(cov: np.ndarray) -> np.ndarray:
    """
    Equal Risk Contribution weights via SLSQP.
    Each asset contributes equally to total portfolio variance.
    Falls back to equal weight if optimisation fails.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    w0 = np.ones(n) / n

    def obj(w):
        var = float(w @ cov @ w)
        if var <= 1e-12:
            return 1e10
        rc = w * (cov @ w) / var   # risk contributions, sum to 1
        return float(np.sum((rc - 1.0 / n) ** 2))

    res = minimize(
        obj, w0, method="SLSQP",
        bounds=[(1e-4, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.maximum(res.x if res.success else w0, 0)
    return w / w.sum()


def _compute_target(
    px: pd.DataFrame,
    i: int,
    tickers: list[str],
    lookback: int,
    skip: int,
    use_erc: bool = False,
    cov_lookback: int = 63,
) -> pd.Series:
    """Compute a single long/short target position vector at bar i."""
    signal_px = px.iloc[i - 1 - skip][tickers]
    past_px   = px.iloc[i - 1 - skip - lookback][tickers]
    signal    = (signal_px - past_px) / past_px

    valid = signal.dropna()
    target = pd.Series(0.0, index=tickers)
    if len(valid) < 4:
        return target

    ranked  = valid.rank(ascending=False)
    n_valid = len(valid)
    q       = max(1, n_valid // 4)
    longs   = valid[ranked <= q].index.tolist()
    shorts  = valid[ranked > n_valid - q].index.tolist()

    if use_erc:
        # Estimate covariance from the last cov_lookback days (no lookahead:
        # uses returns up to and including close[i-1])
        start = max(0, i - cov_lookback)
        ret_hist = px.iloc[start:i].pct_change(fill_method=None).dropna()

        def _book_weights(names: list[str]) -> np.ndarray:
            avail = [t for t in names if t in ret_hist.columns]
            if len(avail) < 2:
                return np.ones(len(names)) / len(names)
            r = ret_hist[avail].dropna(axis=1)
            if r.shape[0] < 5 or r.shape[1] < 2:
                return np.ones(len(names)) / len(names)
            cov = r.cov().values
            # Small diagonal regularisation for numerical stability
            cov += np.eye(len(r.columns)) * cov.trace() / len(r.columns) * 0.01
            return _erc_weights(cov)

        if longs:
            w = _book_weights(longs)
            target[longs] = w
        if shorts:
            w = _book_weights(shorts)
            target[shorts] = -w
    else:
        if longs:
            target[longs]  =  1.0 / len(longs)
        if shorts:
            target[shorts] = -1.0 / len(shorts)

    return target


def _first_trading_days_of_month(dates: list) -> set:
    """Return the set of dates that are the first trading day of their calendar month."""
    dt = pd.to_datetime(dates)
    first_days = set()
    prev_month = None
    for d in dt:
        m = (d.year, d.month)
        if m != prev_month:
            first_days.add(d.normalize())
            prev_month = m
    return first_days


def run_backtest(
    close: pd.DataFrame,
    tickers: list[str],
    lookback: int,
    rebal_days: int,
    skip: int = 0,
    slice_days: int = 1,
    monthly: bool = False,
    use_erc: bool = False,
    cov_lookback: int = 63,
) -> pd.Series:
    """
    Returns a daily P&L series (dollar P&L on $1 long + $1 short gross).

    skip:       trading days to skip at the recent end of the lookback window.
    slice_days: blend last N target portfolios to spread execution over N days.
    monthly:    if True, ignore rebal_days and instead rebalance on the first
                trading day of each calendar month, using the prior month-end
                close as the signal. Overrides rebal_days.
    """
    px = close[tickers].dropna(how="all")

    dates = px.index.tolist()
    if len(dates) < lookback + skip + 2:
        return pd.Series(dtype=float)

    # Pre-compute rebal trigger dates
    if monthly:
        rebal_set = _first_trading_days_of_month(dates)
    else:
        rebal_set = None

    daily_pnl    = []
    positions    = pd.Series(0.0, index=tickers)
    target_queue = deque(maxlen=slice_days)
    rebal_counter = 0

    for i in range(lookback + skip + 1, len(dates)):
        today    = dates[i]
        px_today = px.loc[today, tickers]
        prev_px  = px.iloc[i - 1][tickers]

        # 1. P&L: existing blended positions earn today's return
        day_ret = (px_today - prev_px) / prev_px
        day_pnl = (positions * day_ret).sum()
        daily_pnl.append((today, day_pnl))

        # 2. Determine whether to rebalance end-of-day
        if monthly:
            do_rebal = pd.Timestamp(today).normalize() in rebal_set
        else:
            do_rebal = (rebal_counter == 0)

        if do_rebal:
            target = _compute_target(px, i, tickers, lookback, skip, use_erc, cov_lookback)
            target_queue.append(target)

        # Blended position = equal-weight average of all targets in the queue
        if target_queue:
            positions = sum(target_queue) / len(target_queue)

        if not monthly:
            rebal_counter = (rebal_counter + 1) % rebal_days

    if not daily_pnl:
        return pd.Series(dtype=float)

    return pd.DataFrame(daily_pnl, columns=["date", "pnl"]).set_index("date")["pnl"]


# ── Statistics ─────────────────────────────────────────────────────────────────

def stats(pnl: pd.Series) -> dict:
    if pnl.empty or pnl.std() == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "n_days": 0}
    cum  = (1 + pnl).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    ann  = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    return {
        "ann_ret":  round(float(ann), 4),
        "sharpe":   round(float(ann / vol) if vol > 0 else 0, 3),
        "max_dd":   round(float(dd.min()), 4),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "n_days":   int(len(pnl)),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_cumret(pnl_dict: dict[str, pd.Series], title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (skipping plot — install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, pnl in pnl_dict.items():
        if pnl.empty:
            continue
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum.values * 100, label=label, linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_all_lookbacks(
    results: dict,   # (universe_label, lookback, skip) -> pnl Series
    out_dir: Path,
) -> None:
    """One chart per universe showing all (lookback, skip) combos on same axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    universe_labels = sorted({k[0] for k in results})
    param_combos    = sorted({(k[1], k[2]) for k in results})   # (lookback, skip)

    for ulabel in universe_labels:
        fig, ax = plt.subplots(figsize=(12, 5))
        for lb, sk in param_combos:
            pnl = results.get((ulabel, lb, sk))
            if pnl is None or pnl.empty:
                continue
            cum = (1 + pnl).cumprod() - 1
            label = f"lb={lb}d skip={sk}d" if sk > 0 else f"lb={lb}d"
            ax.plot(pd.to_datetime(cum.index), cum.values * 100,
                    label=label, linewidth=1.2)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Cumulative return (%)")
        ax.set_title(f"Momentum — {ulabel}  (long top Q / short bot Q, weekly rebal)",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"cumret_{ulabel}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved -> {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Momentum backtest on least-correlated ETF universes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lookbacks", type=int, nargs="+", default=DEFAULT_LOOKBACKS,
                        metavar="N", help="Signal lookback periods in trading days")
    parser.add_argument("--rebal", type=int, nargs="+", default=[REBAL_DAYS],
                        metavar="N", help="Rebalancing frequency in trading days (can pass multiple)")
    parser.add_argument("--skip", type=int, nargs="+", default=[0],
                        metavar="N", help="Days to skip at recent end of lookback (0=no skip, 21=1M skip for 12-1 momentum)")
    parser.add_argument("--slice", type=int, nargs="+", default=[1],
                        metavar="N", dest="slice_days",
                        help="Spread each rebal over N days by blending the last N target portfolios (1=no slicing)")
    parser.add_argument("--erc", action="store_true", default=False,
                        help="Use Equal Risk Contribution weighting in long and short books")
    parser.add_argument("--cov-lookback", type=int, default=63,
                        metavar="N", help="Days of return history used to estimate covariance for ERC")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Save cumulative return charts")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--output", default=str(_HERE / "results_momentum"))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load universes ─────────────────────────────────────────────────────────
    print("Loading universes ...")
    universes = load_universes()
    for label, tickers in universes.items():
        print(f"  {label:<25s} {len(tickers):>3} tickers")

    all_tickers = set(t for tickers in universes.values() for t in tickers)
    print(f"\nLoading prices for {len(all_tickers)} unique tickers ...")
    close = load_prices(all_tickers)
    print(f"  {len(close):,} trading days  {close.index[0]} → {close.index[-1]}")

    # ── Run backtests ──────────────────────────────────────────────────────────
    results   = {}   # (universe_label, lookback) -> pnl series
    stats_rows = []

    combos = list(product(
        sorted(universes.keys()),
        sorted(args.lookbacks),
        sorted(args.rebal),
        sorted(args.skip),
        sorted(args.slice_days),
    ))
    print(f"\nRunning {len(combos)} backtests ...\n")

    for universe_label, lookback, rebal, skip, slice_d in combos:
        tickers = universes[universe_label]
        available = [t for t in tickers if t in close.columns]
        missing   = len(tickers) - len(available)

        pnl = run_backtest(close, available, lookback, rebal, skip, slice_d,
                           use_erc=args.erc, cov_lookback=args.cov_lookback)
        results[(universe_label, lookback, skip)] = pnl
        s   = stats(pnl)

        slice_tag = f"  slice={slice_d}d" if slice_d > 1 else ""
        flag = f"  ({missing} missing)" if missing else ""
        print(
            f"  {universe_label:<25s}  lb={lookback:>3}d  skip={skip:>2}d  rebal={rebal:>2}d{slice_tag}  "
            f"Sharpe={s['sharpe']:>6.2f}  "
            f"AnnRet={s['ann_ret']:>+7.1%}  "
            f"MaxDD={s['max_dd']:>7.1%}  "
            f"WinRate={s['win_rate']:.0%}{flag}"
        )

        stats_rows.append({
            "universe":   universe_label,
            "lookback":   lookback,
            "skip_days":  skip,
            "rebal_days": rebal,
            "slice_days": slice_d,
            "n_tickers":  len(available),
            **s,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "stats_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n{'='*80}")
    print("SUMMARY  (sorted by Sharpe)")
    print(f"{'='*80}")
    print(stats_df.sort_values("sharpe", ascending=False).to_string(index=False))
    print(f"\nSaved -> {stats_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if args.plot:
        print("\nSaving charts ...")
        plot_all_lookbacks(results, out_dir)

    print(f"\nAll outputs -> {out_dir}/")


if __name__ == "__main__":
    main()
