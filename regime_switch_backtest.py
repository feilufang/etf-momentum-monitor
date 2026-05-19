#!/usr/bin/env python3
"""
Regime-switching strategy:
  VIX[T-1] < 15  → Momentum  (K100_hierarchical, ERC, 252d lookback, weekly rebal)
  VIX[T-1] > 20  → Reversal  (K100_hierarchical, EW,  5d signal, bottom 25%, daily rebal)
  VIX[T-1] 15–20 → Cash (flat)

Both legs run on the same K100_hierarchical universe.
"""

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
VIX_FILE   = _HERE / "vix_daily.csv"
SPY_FILE   = _HERE / "spy_daily.csv"
OUT_DIR    = _HERE / "results_regime_switch"

LOOKBACK      = 252
REBAL_DAYS    = 5
COV_LB        = 63
SIGNAL_DAYS   = 5
BOTTOM_PCT    = 0.25
VIX_MOM_MAX   = 15    # momentum active when VIX < this
VIX_REV_MIN   = 20    # reversal active when VIX > this


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    return pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()


def load_prices(tickers: list[str]) -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
    daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
    close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    return close.sort_index().ffill()


def load_vix() -> pd.Series:
    vix = pd.read_csv(VIX_FILE, index_col=0, parse_dates=True).squeeze()
    vix.index = vix.index.strftime("%Y-%m-%d")
    return vix.sort_index()


def load_spy() -> pd.Series:
    spy = pd.read_csv(SPY_FILE, index_col=0, parse_dates=True).squeeze()
    spy.index = spy.index.strftime("%Y-%m-%d")
    return spy.sort_index().pct_change(fill_method=None)  # daily returns


def erc_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    w0 = np.ones(n) / n
    def obj(w):
        var = float(w @ cov @ w)
        if var <= 1e-12: return 1e10
        return float(np.sum((w * (cov @ w) / var - 1 / n) ** 2))
    res = minimize(obj, w0, method="SLSQP",
                   bounds=[(1e-4, 1.0)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"ftol": 1e-10, "maxiter": 500})
    w = np.maximum(res.x if res.success else w0, 0)
    return w / w.sum()


def erc_book(members: list[str], daily_ret: pd.DataFrame, i: int) -> np.ndarray:
    if len(members) == 1:
        return np.array([1.0])
    hist = daily_ret[members].iloc[max(0, i - COV_LB):i].dropna()
    if len(hist) < 10:
        return np.ones(len(members)) / len(members)
    cov = hist.cov().values
    cov += np.eye(len(members)) * np.diag(cov).mean() * 0.10
    return erc_weights(cov)


def stats(pnl: pd.Series) -> dict:
    if len(pnl) < 5:
        return dict(n=len(pnl), ann_ret=0, sharpe=0, max_dd=0, win_rate=0, cum_ret=0)
    ann_ret = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum     = (1 + pnl).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(n=len(pnl), ann_ret=ann_ret, sharpe=sharpe,
                max_dd=max_dd, win_rate=(pnl > 0).mean(), cum_ret=cum.iloc[-1] - 1)


# ── Momentum engine (ERC, weekly rebal) ───────────────────────────────────────

def run_momentum(px: pd.DataFrame, avail: list[str], daily_ret: pd.DataFrame) -> pd.Series:
    dates     = px.index.tolist()
    positions = pd.Series(0.0, index=avail)
    rebal_ctr = 0
    pnl_vals, pnl_idx = [], []

    for i in range(LOOKBACK + 1, len(dates)):
        dr = daily_ret.iloc[i][avail]
        pnl_vals.append(float((positions * dr).sum()))
        pnl_idx.append(dates[i])

        if rebal_ctr == 0:
            signal = ((px.iloc[i - 1][avail] - px.iloc[i - 1 - LOOKBACK][avail])
                      / px.iloc[i - 1 - LOOKBACK][avail]).dropna()
            if len(signal) >= 4:
                ranked = signal.rank(ascending=False)
                n_v    = len(signal)
                q      = max(1, n_v // 4)
                longs  = signal[ranked <= q].index.tolist()
                shorts = signal[ranked > n_v - q].index.tolist()
                positions = pd.Series(0.0, index=avail)
                wl = erc_book(longs,  daily_ret, i)
                ws = erc_book(shorts, daily_ret, i)
                for t, w in zip(longs,  wl): positions[t] =  w
                for t, w in zip(shorts, ws): positions[t] = -w

        rebal_ctr = (rebal_ctr + 1) % REBAL_DAYS

    return pd.Series(pnl_vals, index=pnl_idx, name="momentum")


# ── Reversal engine (EW, bottom 25%, daily rebal) ─────────────────────────────

def run_reversal(px: pd.DataFrame, avail: list[str], daily_ret: pd.DataFrame) -> pd.Series:
    signal    = px[avail].pct_change(periods=SIGNAL_DAYS, fill_method=None)
    n_select  = max(1, int(len(avail) * BOTTOM_PCT))
    pnl_vals, pnl_idx = [], []

    for i in range(SIGNAL_DAYS + 1, len(px)):
        sig      = signal.iloc[i - 1].dropna()
        if len(sig) < n_select:
            continue
        selected = sig.nsmallest(n_select).index.tolist()
        day_rets = daily_ret.iloc[i][selected].dropna()
        if day_rets.empty:
            continue
        pnl_vals.append(day_rets.mean())
        pnl_idx.append(px.index[i])

    return pd.Series(pnl_vals, index=pnl_idx, name="reversal")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    tickers   = load_universe()
    close     = load_prices(tickers)
    vix       = load_vix()
    spy_ret   = load_spy()
    avail     = [t for t in tickers if t in close.columns]
    daily_ret = close[avail].pct_change(fill_method=None)

    print(f"  {len(avail)} tickers  |  {len(close)} days  "
          f"{close.index[0]} → {close.index[-1]}")

    # Align VIX to price index (T-1, no lookahead)
    vix_t1 = vix.reindex(close.index).ffill().shift(1)

    print("\nRunning momentum leg ...")
    mom_pnl = run_momentum(close[avail], avail, daily_ret)

    print("Running reversal leg (unhedged) ...")
    rev_pnl_raw = run_reversal(close[avail], avail, daily_ret)

    # Hedge reversal with short SPY (equal notional)
    # rev_hedged[T] = long bottom-25% return - SPY return
    spy_aligned  = spy_ret.reindex(rev_pnl_raw.index).fillna(0)
    rev_pnl      = rev_pnl_raw - spy_aligned
    rev_pnl.name = "reversal_hedged"

    # Align both to a common index
    common = mom_pnl.index.intersection(rev_pnl.index)
    mom_pnl     = mom_pnl.reindex(common)
    rev_pnl     = rev_pnl.reindex(common)
    rev_pnl_raw = rev_pnl_raw.reindex(common)
    vix_reg     = vix_t1.reindex(common)

    # Regime labels
    regime = pd.Series("cash", index=common)
    regime[vix_reg <  VIX_MOM_MAX] = "momentum"
    regime[vix_reg >  VIX_REV_MIN] = "reversal"

    # Combined P&L: select leg by regime, 0 in cash
    combined = pd.Series(0.0, index=common)
    combined[regime == "momentum"] = mom_pnl[regime == "momentum"]
    combined[regime == "reversal"] = rev_pnl[regime == "reversal"]

    # ── Stats ──────────────────────────────────────────────────────────────────
    print(f"\nRegime breakdown:")
    for r in ["momentum", "reversal", "cash"]:
        n = (regime == r).sum()
        print(f"  {r:<10}  {n:>5} days  ({n/len(regime)*100:.0f}%)")

    print()
    rows = [
        ("Combined",              combined),
        ("Momentum (all)",        mom_pnl),
        ("Reversal hedged (all)", rev_pnl),
        ("Reversal raw (all)",    rev_pnl_raw),
        ("Mom VIX<15 only",       mom_pnl[regime == "momentum"]),
        ("Rev hedged VIX>20",     rev_pnl[regime == "reversal"]),
        ("Rev raw VIX>20",        rev_pnl_raw[regime == "reversal"]),
    ]

    hdr = f"{'Strategy':<20}  {'Days':>6}  {'Ann Ret':>9}  {'Sharpe':>7}  {'Max DD':>8}  {'WinRate':>8}  {'Cum Ret':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, p in rows:
        s = stats(p.dropna())
        print(f"{label:<20}  {s['n']:>6}  {s['ann_ret']*100:>+8.1f}%  "
              f"{s['sharpe']:>+7.2f}  {s['max_dd']*100:>+8.1f}%  "
              f"{s['win_rate']*100:>7.0f}%  {s['cum_ret']*100:>+8.1f}%")

    # ── Chart ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
                                        sharex=True)

        # Shade regime backgrounds on top panel
        regime_colors = {"momentum": "#E3F2FD", "reversal": "#FCE4EC", "cash": "#F5F5F5"}
        dates_dt = pd.to_datetime(common)
        prev_r, start_i = regime.iloc[0], 0
        for i in range(1, len(regime)):
            if regime.iloc[i] != prev_r or i == len(regime) - 1:
                end_i = i if regime.iloc[i] != prev_r else i + 1
                ax1.axvspan(dates_dt[start_i], dates_dt[min(end_i, len(dates_dt)-1)],
                            color=regime_colors[prev_r], alpha=0.6, linewidth=0)
                start_i, prev_r = i, regime.iloc[i]

        # Cumulative return lines
        for label, p, color, lw, ls in [
            ("Combined (regime switch)",   combined,    "#212121", 2.2, "-"),
            ("Momentum (all days)",        mom_pnl,     "#1E88E5", 1.2, "--"),
            ("Reversal hedged (all days)", rev_pnl,     "#E53935", 1.2, "--"),
            ("Reversal raw (all days)",    rev_pnl_raw, "#E53935", 1.0, ":"),
        ]:
            cum = (1 + p.fillna(0)).cumprod() - 1
            ax1.plot(pd.to_datetime(cum.index), cum.values * 100,
                     label=label, color=color, linewidth=lw, linestyle=ls)

        ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax1.set_ylabel("Cumulative return (%)")
        ax1.set_title(
            f"Regime-switching strategy  |  K100_hierarchical\n"
            f"VIX<{VIX_MOM_MAX} → Momentum (ERC, weekly)   "
            f"VIX>{VIX_REV_MIN} → Reversal (EW, bottom 25%, daily)   "
            f"VIX {VIX_MOM_MAX}–{VIX_REV_MIN} → Cash",
            fontsize=10,
        )
        legend_handles = [
            plt.Line2D([0], [0], color="#212121", lw=2.2, label="Combined (regime switch)"),
            plt.Line2D([0], [0], color="#1E88E5", lw=1.2, ls="--", label="Momentum (all days)"),
            plt.Line2D([0], [0], color="#E53935", lw=1.2, ls="--", label="Reversal hedged (all days)"),
            plt.Line2D([0], [0], color="#E53935", lw=1.0, ls=":",  label="Reversal raw (all days)"),
            mpatches.Patch(color="#E3F2FD", alpha=0.8, label=f"VIX<{VIX_MOM_MAX} (momentum)"),
            mpatches.Patch(color="#FCE4EC", alpha=0.8, label=f"VIX>{VIX_REV_MIN} (reversal)"),
            mpatches.Patch(color="#F5F5F5", alpha=0.8, label="VIX 15–20 (cash)"),
        ]
        ax1.legend(handles=legend_handles, fontsize=8, ncol=3, loc="upper left")
        ax1.grid(alpha=0.25)

        # Bottom panel: VIX time series with threshold bands
        vix_plot = vix.reindex(pd.to_datetime(common).strftime("%Y-%m-%d")).ffill()
        ax2.plot(dates_dt, vix_plot.values, color="#555", linewidth=1.0, label="VIX")
        ax2.axhline(VIX_MOM_MAX, color="#1E88E5", linewidth=1.2, linestyle="--",
                    label=f"VIX={VIX_MOM_MAX}")
        ax2.axhline(VIX_REV_MIN, color="#E53935", linewidth=1.2, linestyle="--",
                    label=f"VIX={VIX_REV_MIN}")
        ax2.fill_between(dates_dt, 0, vix_plot.values,
                         where=(vix_plot.values < VIX_MOM_MAX),
                         color="#1E88E5", alpha=0.2)
        ax2.fill_between(dates_dt, 0, vix_plot.values,
                         where=(vix_plot.values > VIX_REV_MIN),
                         color="#E53935", alpha=0.2)
        ax2.set_ylabel("VIX")
        ax2.set_ylim(0, vix_plot.max() * 1.05)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.25)

        fig.tight_layout()
        out = OUT_DIR / "regime_switch_cumret.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\nChart -> {out}")

    except ImportError:
        print("matplotlib not available")

    # Save daily P&L
    out_df = pd.DataFrame({
        "regime":          regime,
        "combined":        combined,
        "momentum":        mom_pnl,
        "reversal_hedged": rev_pnl,
        "reversal_raw":    rev_pnl_raw,
        "vix_t1":          vix_reg,
    })
    out_df.to_csv(OUT_DIR / "regime_switch_pnl.csv")
    print(f"P&L CSV -> {OUT_DIR / 'regime_switch_pnl.csv'}")


if __name__ == "__main__":
    main()
