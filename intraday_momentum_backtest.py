#!/usr/bin/env python3
"""
Intraday-signal momentum backtest on the least-correlated ETF universes.

Signal   : Intraday return from previous close to 3:30 PM ET today.
           Observed at 3:30 PM — 30 min before the close, so MOC orders
           can still be placed.  No look-ahead.

Portfolio: Long top quartile, short bottom quartile. Equal weight.
           Dollar-neutral: long $1, short $1 per side.

Execution: Today's closing price (same day as the signal).

Exit     : Positions held and exited at closing prices over REBAL_DAYS.

Inputs
------
    results_corr/selected_K*_*.csv   from select_uncorrelated_etfs.py
    data/minute_etf/<date>.parquet   3:30 PM intraday prices
    data/daily/all.parquet           previous close + execution close

Outputs (in results_intraday/)
-------------------------------
    stats_summary.csv
    cumret_<universe>.png

Usage
-----
    python intraday_momentum_backtest.py
    python intraday_momentum_backtest.py --rebal 1 3 5 10
    python intraday_momentum_backtest.py --signal-time 15 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DATA_DIR   = _HERE.parent.parent / "data"
DAILY_FILE = DATA_DIR / "daily/all.parquet"
MINUTE_DIR = DATA_DIR / "minute_etf"
CORR_DIR   = _HERE / "results_corr"


# ── Load universe lists ────────────────────────────────────────────────────────

def load_universes() -> dict[str, list[str]]:
    universes = {}
    for path in sorted(CORR_DIR.glob("selected_K*_*.csv")):
        label = path.stem.replace("selected_", "")
        universes[label] = pd.read_csv(path)["ticker"].tolist()
    if not universes:
        raise FileNotFoundError(f"No selected_K*.csv in {CORR_DIR}. Run select_uncorrelated_etfs.py first.")
    return universes


# ── Pre-load all 3:30 PM bars ─────────────────────────────────────────────────

def load_intraday_prices(tickers: set[str], sig_hour: int, sig_minute: int) -> pd.DataFrame:
    """
    Returns a (date × ticker) DataFrame of intraday prices at the signal time.
    Only loads dates that have a minute file.
    """
    files = sorted(MINUTE_DIR.glob("*.parquet"))
    print(f"  Loading {len(files)} minute files for {sig_hour:02d}:{sig_minute:02d} ET bar ...")

    rows = []
    for f in files:
        date_str = f.stem
        try:
            df = pd.read_parquet(f, columns=["ticker", "timestamp_ms", "close"])
        except Exception:
            continue
        df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        dt_et = (pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                   .dt.tz_convert("America/New_York"))
        mask = (dt_et.dt.hour == sig_hour) & (dt_et.dt.minute == sig_minute)
        bar = df[mask]
        if bar.empty:
            continue
        bar = bar.sort_values("timestamp_ms").groupby("ticker")["close"].last()
        for ticker, px in bar.items():
            rows.append({"date": date_str, "ticker": ticker, "px": px})

    if not rows:
        raise RuntimeError("No intraday bars found. Check minute data path and signal time.")

    intra = pd.DataFrame(rows).pivot_table(
        index="date", columns="ticker", values="px", aggfunc="last"
    )
    print(f"  {len(intra):,} dates, {len(intra.columns):,} tickers with {sig_hour:02d}:{sig_minute:02d} bar")
    return intra


# ── Load daily closes ──────────────────────────────────────────────────────────

def load_daily_closes(tickers: set[str]) -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
    daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
    close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    close.sort_index(inplace=True)
    close = close.ffill()
    return close


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(
    intra: pd.DataFrame,    # (date × ticker) 3:30 PM prices
    close: pd.DataFrame,    # (date × ticker) daily closes
    tickers: list[str],
    rebal_days: int,
) -> pd.Series:
    """
    Signal  : (intra[t] - close[t-1]) / close[t-1]   — intraday return to signal time
    Executed: at close[t]  (same day, MOC order)
    Exit    : close[t+1], close[t+2], ... (rebal_days slices, equal weight)

    Returns daily P&L series on $1 long / $1 short gross.
    """
    # Restrict to dates that exist in BOTH intraday and daily data
    common_dates = sorted(set(intra.index) & set(close.index))
    if len(common_dates) < 5:
        return pd.Series(dtype=float)

    # Align tickers to those available in both datasets
    avail = [t for t in tickers if t in intra.columns and t in close.columns]
    if len(avail) < 4:
        return pd.Series(dtype=float)

    intra_px = intra[avail].reindex(common_dates)
    close_px = close[avail].reindex(common_dates).ffill()

    all_close_dates = close.index.tolist()
    n_q = max(1, len(avail) // 4)

    # Positions entered on signal date (keyed by signal date index in common_dates)
    # Each entry: (entry_close_px series, weight series, days_remaining)
    open_positions: list[tuple[pd.Series, pd.Series, int]] = []

    rebal_counter = 0
    daily_pnl = []

    for i, date in enumerate(common_dates):
        prev_close_date = common_dates[i - 1] if i > 0 else None
        if prev_close_date is None:
            continue

        # ── Settle open positions exiting today ────────────────────────────────
        today_close = close_px.loc[date]
        day_pnl = 0.0
        still_open = []
        for entry_px, weights, days_left in open_positions:
            # Each position exits 1/rebal_days per day
            slice_ret = ((today_close - entry_px) / entry_px * weights).sum() / rebal_days
            day_pnl += slice_ret
            if days_left > 1:
                still_open.append((entry_px, weights, days_left - 1))
        open_positions = still_open
        daily_pnl.append((date, day_pnl))

        # ── New signal: rebalance ──────────────────────────────────────────────
        if rebal_counter == 0:
            prev_close = close_px.loc[prev_close_date]
            intra_now  = intra_px.loc[date]

            # Signal = intraday return to 3:30 PM
            sig = ((intra_now - prev_close) / prev_close).dropna()
            if len(sig) < 4:
                rebal_counter = (rebal_counter + 1) % rebal_days
                continue

            n_v  = len(sig)
            q    = max(1, n_v // 4)
            ranked = sig.rank(ascending=False)
            longs  = sig[ranked <= q].index.tolist()
            shorts = sig[ranked > n_v - q].index.tolist()

            weights = pd.Series(0.0, index=avail)
            if longs:  weights[longs]  =  1.0 / len(longs)
            if shorts: weights[shorts] = -1.0 / len(shorts)

            entry_close = close_px.loc[date]
            open_positions.append((entry_close, weights, rebal_days))

        rebal_counter = (rebal_counter + 1) % rebal_days

    if not daily_pnl:
        return pd.Series(dtype=float)
    return pd.DataFrame(daily_pnl, columns=["date", "pnl"]).set_index("date")["pnl"]


# ── Statistics ─────────────────────────────────────────────────────────────────

def stats(pnl: pd.Series) -> dict:
    if pnl.empty or pnl.std() == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "n_days": 0}
    cum  = (1 + pnl).cumprod()
    dd   = (cum - cum.cummax()) / cum.cummax()
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

def plot_all(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rebal_vals   = sorted({k[1] for k in results})
    univ_labels  = sorted({k[0] for k in results})

    for ulabel in univ_labels:
        fig, ax = plt.subplots(figsize=(12, 5))
        for rb in rebal_vals:
            pnl = results.get((ulabel, rb))
            if pnl is None or pnl.empty:
                continue
            cum = (1 + pnl).cumprod() - 1
            ax.plot(pd.to_datetime(cum.index), cum.values * 100,
                    label=f"rebal={rb}d", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Cumulative return (%)")
        ax.set_title(f"Intraday momentum (3:30 PM signal → close trade) — {ulabel}", fontsize=11)
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
        description="Intraday-signal (3:30 PM → close) momentum backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rebal", type=int, nargs="+", default=[1, 3, 5, 10],
                        metavar="N", help="Holding / rebalancing period in trading days")
    parser.add_argument("--signal-time", type=int, nargs=2, default=[15, 30],
                        metavar=("HH", "MM"), help="Intraday signal bar time (ET, 24h)")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--output", default=str(_HERE / "results_intraday"))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_hour, sig_minute = args.signal_time

    # ── Load universes ─────────────────────────────────────────────────────────
    print("Loading universes ...")
    universes = load_universes()
    for label, tickers in universes.items():
        print(f"  {label:<25s} {len(tickers):>3} tickers")

    all_tickers = set(t for ts in universes.values() for t in ts)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading intraday prices ({sig_hour:02d}:{sig_minute:02d} ET) ...")
    intra = load_intraday_prices(all_tickers, sig_hour, sig_minute)

    print("\nLoading daily closes ...")
    close = load_daily_closes(all_tickers)
    print(f"  {len(close):,} trading days  {close.index[0]} → {close.index[-1]}")

    # ── Run backtests ──────────────────────────────────────────────────────────
    results    = {}
    stats_rows = []

    combos = [(ul, rb) for ul in sorted(universes) for rb in sorted(args.rebal)]
    print(f"\nRunning {len(combos)} backtests ...\n")

    for universe_label, rebal in combos:
        tickers = universes[universe_label]
        pnl = run_backtest(intra, close, tickers, rebal)
        results[(universe_label, rebal)] = pnl
        s   = stats(pnl)
        print(
            f"  {universe_label:<25s}  rebal={rebal:>2}d  "
            f"Sharpe={s['sharpe']:>6.2f}  "
            f"AnnRet={s['ann_ret']:>+7.1%}  "
            f"MaxDD={s['max_dd']:>7.1%}  "
            f"WinRate={s['win_rate']:.0%}  "
            f"n={s['n_days']}"
        )
        stats_rows.append({"universe": universe_label, "rebal_days": rebal, **s})

    # ── Summary ────────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "stats_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n{'='*75}")
    print("SUMMARY  (sorted by Sharpe)")
    print(f"{'='*75}")
    print(stats_df.sort_values("sharpe", ascending=False).to_string(index=False))
    print(f"\nSaved -> {stats_path}")

    if args.plot:
        print("\nSaving charts ...")
        plot_all(results, out_dir)

    print(f"\nAll outputs -> {out_dir}/")


if __name__ == "__main__":
    main()
