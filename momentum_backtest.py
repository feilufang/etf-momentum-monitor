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
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

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

def run_backtest(
    close: pd.DataFrame,
    tickers: list[str],
    lookback: int,
    rebal_days: int,
    skip: int = 0,
) -> pd.Series:
    """
    Returns a daily P&L series (dollar P&L on $1 long + $1 short gross).

    skip: trading days to exclude from the recent end of the lookback window.
          Signal = (close[i-1-skip] - close[i-1-skip-lookback]) / close[i-1-skip-lookback]
          skip=0 is the standard signal; skip=5 implements "12-1 week" momentum.
    """
    px = close[tickers].dropna(how="all")

    # Need enough history before first signal
    dates = px.index.tolist()
    if len(dates) < lookback + skip + rebal_days + 1:
        return pd.Series(dtype=float)

    n = len(tickers)
    q_size = max(1, n // 4)     # quartile size

    daily_pnl = []
    positions = pd.Series(0.0, index=tickers)   # current position (signed notional)

    rebal_counter = 0
    for i in range(lookback + skip + 1, len(dates)):
        today    = dates[i]
        px_today = px.loc[today, tickers]
        prev_px  = px.iloc[i - 1][tickers]

        # 1. P&L first — old positions earn today's return (close[i-1] -> close[i])
        #    These positions were entered at yesterday's close.
        day_ret = (px_today - prev_px) / prev_px
        day_pnl = (positions * day_ret).sum()
        daily_pnl.append((today, day_pnl))

        # 2. End-of-day: update positions using a lagged signal.
        #    Signal window ends skip days ago to avoid short-term reversal noise.
        #    New positions enter at today's close (close[i]) — no look-ahead.
        if rebal_counter == 0:
            signal_px = px.iloc[i - 1 - skip][tickers]              # close[i-1-skip]
            past_px   = px.iloc[i - 1 - skip - lookback][tickers]   # close[i-1-skip-LB]
            signal    = (signal_px - past_px) / past_px

            valid = signal.dropna()
            if len(valid) < 4:
                positions = pd.Series(0.0, index=tickers)
            else:
                ranked  = valid.rank(ascending=False)
                n_valid = len(valid)
                q       = max(1, n_valid // 4)
                longs   = valid[ranked <= q].index.tolist()
                shorts  = valid[ranked > n_valid - q].index.tolist()
                positions = pd.Series(0.0, index=tickers)
                if longs:
                    positions[longs]  =  1.0 / len(longs)
                if shorts:
                    positions[shorts] = -1.0 / len(shorts)

        rebal_counter = (rebal_counter + 1) % rebal_days

    if not daily_pnl:
        return pd.Series(dtype=float)

    pnl = pd.DataFrame(daily_pnl, columns=["date", "pnl"]).set_index("date")["pnl"]
    return pnl


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
                        metavar="N", help="Days to skip at recent end of lookback (0=no skip, 5=1wk, 10=2wk...)")
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

    combos = list(product(sorted(universes.keys()), sorted(args.lookbacks), sorted(args.rebal), sorted(args.skip)))
    print(f"\nRunning {len(combos)} backtests ...\n")

    for universe_label, lookback, rebal, skip in combos:
        tickers = universes[universe_label]
        available = [t for t in tickers if t in close.columns]
        missing   = len(tickers) - len(available)

        pnl = run_backtest(close, available, lookback, rebal, skip)
        results[(universe_label, lookback, skip)] = pnl
        s   = stats(pnl)

        flag = f"  ({missing} missing)" if missing else ""
        print(
            f"  {universe_label:<25s}  lb={lookback:>3}d  skip={skip:>2}d  rebal={rebal:>2}d  "
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
