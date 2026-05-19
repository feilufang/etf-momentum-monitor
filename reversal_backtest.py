#!/usr/bin/env python3
"""
Short-term mean-reversion backtest on K100_hierarchical universe.

Signal  : 5-day lookback return (signal computed from T-1 prices, no lookahead)
Long    : Bottom N% by 5-day return — equal weight or ERC, $1 gross invested
Rebal   : Daily
Thresholds: 5%, 10%, 25% of universe

Usage
-----
    python reversal_backtest.py
    python reversal_backtest.py --signal-days 3 --universe K60_hierarchical
    python reversal_backtest.py --erc
"""

import argparse
import sys
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
OUT_DIR    = _HERE / "results_reversal"
VIX_FILE   = _HERE / "vix_daily.csv"


def load_universe(name: str) -> list[str]:
    path = CORR_DIR / f"selected_{name}.csv"
    return pd.read_csv(path)["ticker"].tolist()


def load_prices(tickers: list[str]) -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
    daily = daily[daily["ticker"].isin(tickers)]
    close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def load_vix() -> pd.Series:
    vix = pd.read_csv(VIX_FILE, index_col=0, parse_dates=True).squeeze()
    vix.name = "vix"
    return vix.sort_index()


def _erc_weights(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution weights via SLSQP. Falls back to equal weight on failure."""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    w0 = np.ones(n) / n

    def obj(w):
        var = float(w @ cov @ w)
        if var <= 1e-12:
            return 1e10
        rc = w * (cov @ w) / var
        return float(np.sum((rc - 1.0 / n) ** 2))

    res = minimize(
        obj, w0, method="SLSQP",
        bounds=[(1e-4, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.maximum(res.x if res.success else w0, 0)
    return w / w.sum()


def run_reversal(close: pd.DataFrame, tickers: list[str],
                 signal_days: int, bottom_pct: float,
                 use_erc: bool = False, cov_lookback: int = 63,
                 vix: pd.Series | None = None,
                 vix_threshold: float | None = None,
                 vix_above: bool = True) -> pd.Series:
    """
    Each day T:
      signal   = N-day return ending at T-1 close
      selected = bottom_pct fraction of universe by that signal
      weights  = equal weight (default) or ERC using cov_lookback days of returns
      filter   = if vix_above=True,  skip day when vix[T-1] <= threshold
                 if vix_above=False, skip day when vix[T-1] >= threshold
      pnl[T]   = weighted avg daily return (close[T] / close[T-1] - 1) of selected
    """
    avail = [t for t in tickers if t in close.columns]
    px    = close[avail].copy()

    daily_ret = px.pct_change(fill_method=None)
    signal    = px.pct_change(periods=signal_days, fill_method=None)

    # Align VIX to price index, forward-fill gaps (holidays), shift by 1 (use T-1 value)
    vix_aligned = None
    if vix is not None and vix_threshold is not None:
        vix_aligned = vix.reindex(px.index).ffill().shift(1)

    n_select = max(1, int(len(avail) * bottom_pct))
    warmup   = max(signal_days + 1, cov_lookback + 1) if use_erc else signal_days + 1

    pnl_vals = []
    pnl_idx  = []

    for i in range(warmup, len(px)):
        # VIX filter: skip based on direction flag
        if vix_aligned is not None:
            v = vix_aligned.iloc[i]
            if pd.isna(v):
                continue
            if vix_above and v <= vix_threshold:
                continue
            if not vix_above and v >= vix_threshold:
                continue

        sig = signal.iloc[i - 1].dropna()
        if len(sig) < n_select:
            continue

        selected = sig.nsmallest(n_select).index.tolist()
        day_rets = daily_ret.iloc[i][selected].dropna()
        if day_rets.empty:
            continue

        sel = day_rets.index.tolist()

        if use_erc and len(sel) > 1:
            hist = daily_ret[sel].iloc[max(0, i - cov_lookback): i].dropna()
            if len(hist) >= 10:
                cov = hist.cov().values
                cov += np.eye(len(sel)) * np.diag(cov).mean() * 0.10
                w = _erc_weights(cov)
            else:
                w = np.ones(len(sel)) / len(sel)
        else:
            w = np.ones(len(sel)) / len(sel)

        pnl_vals.append(float(w @ day_rets.values))
        pnl_idx.append(px.index[i])

    return pd.Series(pnl_vals, index=pnl_idx, name=f"bot{bottom_pct:.0%}")


def stats(pnl: pd.Series) -> dict:
    ann      = 252
    ann_ret  = pnl.mean() * ann
    ann_vol  = pnl.std() * np.sqrt(ann)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum      = (1 + pnl).cumprod()
    roll_max = cum.cummax()
    max_dd   = ((cum - roll_max) / roll_max).min()
    win_rate = (pnl > 0).mean()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=max_dd, win_rate=win_rate, n_days=len(pnl))


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--universe",    default="K100_hierarchical")
    parser.add_argument("--signal-days", type=int, default=5)
    parser.add_argument("--thresholds",  type=float, nargs="+",
                        default=[0.05, 0.10, 0.25],
                        metavar="P", help="Bottom fractions to test (e.g. 0.05 0.10 0.25)")
    parser.add_argument("--cov-lookback", type=int, default=63,
                        help="Days of return history for ERC covariance estimate")
    parser.add_argument("--vix-threshold", type=float, default=20.0,
                        help="VIX level to filter on (0 = always trade)")
    parser.add_argument("--vix-below", action="store_true", default=False,
                        help="Trade when VIX < threshold (default: trade when VIX > threshold)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Universe : {args.universe}")
    tickers = load_universe(args.universe)
    print(f"  {len(tickers)} tickers")

    print("Loading prices ...")
    close = load_prices(tickers)
    print(f"  {len(close):,} days  {close.index[0].date()} → {close.index[-1].date()}")

    print("Loading VIX ...")
    vix = load_vix()
    print(f"  VIX: {len(vix)} days  {vix.index[0].date()} → {vix.index[-1].date()}")

    vix_thr    = args.vix_threshold if args.vix_threshold > 0 else None
    vix_above  = not args.vix_below
    if vix_thr:
        op = "<" if args.vix_below else ">"
        filter_label = f"VIX{op}{args.vix_threshold:.0f}"
    else:
        filter_label = "no filter"
    print(f"  VIX filter: {filter_label}\n")

    # Run EW: unfiltered baseline + VIX-filtered
    results    = {}   # (pct_label, variant) -> pnl series
    stats_rows = []

    variants = [
        ("All days",       None,    None),
        (filter_label,     vix,     vix_thr),
    ]

    for variant_label, v_series, v_thr in variants:
        print(f"--- EW  [{variant_label}] ---")
        for pct in sorted(args.thresholds):
            n_sel     = max(1, int(len(tickers) * pct))
            pct_label = f"{pct:.0%}"
            print(f"  Bottom {pct_label:>4s} ({n_sel:>3} ETFs) ... ", end="", flush=True)

            pnl = run_reversal(close, tickers, args.signal_days, pct,
                               use_erc=False, cov_lookback=args.cov_lookback,
                               vix=v_series, vix_threshold=v_thr,
                               vix_above=vix_above)
            s   = stats(pnl)
            results[(pct_label, variant_label)] = pnl

            active_days = len(pnl)
            print(f"Sharpe={s['sharpe']:+.2f}  AnnRet={s['ann_ret']*100:+.1f}%  "
                  f"MaxDD={s['max_dd']*100:+.1f}%  WinRate={s['win_rate']*100:.0f}%  "
                  f"ActiveDays={active_days}")

            stats_rows.append({"variant": variant_label, "bottom_pct": pct_label,
                                "n_selected": n_sel, "active_days": active_days, **s})
        print()

    # ── Chart ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping chart")
        return

    palette = {"5%": "#E53935", "10%": "#FB8C00", "25%": "#43A047"}
    style   = {"All days": ("-", 1.4), filter_label: ("--", 2.0)}

    fig, ax = plt.subplots(figsize=(14, 5))

    for (pct_label, variant_label), pnl in results.items():
        n_sel  = max(1, int(len(tickers) * float(pct_label.strip("%")) / 100))
        cum    = (1 + pnl).cumprod() - 1
        ls, lw = style[variant_label]
        color  = palette[pct_label]
        ax.plot(cum.index, cum.values * 100,
                label=f"Bottom {pct_label} ({n_sel} ETFs) [{variant_label}]",
                color=color, linestyle=ls, linewidth=lw)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        f"Short-term Reversal — {args.universe}  "
        f"(long bottom N% by {args.signal_days}-day return, EW | solid=all days  dashed={filter_label})",
        fontsize=11,
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    suffix     = f"vix{'below' if args.vix_below else 'above'}{args.vix_threshold:.0f}" if vix_thr else "nofilter"
    chart_path = OUT_DIR / f"cumret_reversal_{args.universe}_{args.signal_days}d_{suffix}.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"  Chart -> {chart_path}")

    # ── Stats table ────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stats_rows)
    csv_path = OUT_DIR / f"stats_reversal_{args.universe}_{args.signal_days}d_{suffix}.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"  Stats -> {csv_path}")

    print(f"\nSummary (EW, all days vs {filter_label}):")
    print(f"  {'Variant':<14}  {'Bottom':>6}  {'N':>4}  {'ActiveDays':>10}  "
          f"{'AnnRet':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'WinRate':>8}")
    for r in stats_rows:
        print(f"  {r['variant']:<14}  {r['bottom_pct']:>6}  {r['n_selected']:>4}  "
              f"{r['active_days']:>10}  "
              f"{r['ann_ret']*100:>+7.1f}%  {r['sharpe']:>+7.2f}  "
              f"{r['max_dd']*100:>+7.1f}%  {r['win_rate']*100:>7.0f}%")


if __name__ == "__main__":
    main()
