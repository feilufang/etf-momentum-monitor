#!/usr/bin/env python3
"""
Compare reversal backtest: raw 5d return vs 5d return / 252d vol ranking.
Both: VIX>20, EW, daily rebal, K100_hierarchical, SPY-hedged.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DAILY_FILE = _HERE.parent.parent / "data" / "daily" / "all.parquet"
SIGNAL_DAYS = 5
VOL_LB      = 252
BOTTOM_PCT  = 0.25
VIX_MIN     = 20

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()
daily   = pd.read_parquet(DAILY_FILE, columns=["date","ticker","close"])
daily   = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
close   = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().ffill()
avail   = [t for t in tickers if t in close.columns]
dr      = close[avail].pct_change(fill_method=None)

vix = pd.read_csv(_HERE / "vix_daily.csv", index_col=0, parse_dates=True).squeeze()
vix.index = vix.index.strftime("%Y-%m-%d")
vix_t1 = vix.reindex(close.index).ffill().shift(1)

spy = pd.read_csv(_HERE / "spy_daily.csv", index_col=0, parse_dates=True).squeeze()
spy.index = spy.index.strftime("%Y-%m-%d")
spy_ret = spy.pct_change(fill_method=None).reindex(close.index).fillna(0)

n_select = max(1, int(len(avail) * BOTTOM_PCT))
rev_raw  = close[avail].pct_change(periods=SIGNAL_DAYS, fill_method=None)

def run(use_vol_adj: bool):
    pnl_vals, pnl_idx = [], []
    for i in range(VOL_LB + 1, len(close)):
        date = close.index[i]
        v = vix_t1.loc[date] if date in vix_t1.index else np.nan
        if pd.isna(v) or v <= VIX_MIN:
            continue

        sig_5d = rev_raw.iloc[i - 1].dropna()
        if len(sig_5d) < n_select:
            continue

        if use_vol_adj:
            vol_252 = dr[avail].iloc[i - VOL_LB:i].std() * np.sqrt(252)
            signal  = (sig_5d / vol_252.reindex(sig_5d.index).replace(0, np.nan)).dropna()
        else:
            signal = sig_5d

        if len(signal) < n_select:
            continue

        selected = signal.nsmallest(n_select).index.tolist()
        day_rets = dr.iloc[i][selected].dropna()
        if day_rets.empty:
            continue

        spy_d = spy_ret.loc[date] if date in spy_ret.index else 0.0
        pnl_vals.append(day_rets.mean() - spy_d)
        pnl_idx.append(date)

    return pd.Series(pnl_vals, index=pnl_idx)

print("Running raw 5d signal ...")
pnl_raw = run(use_vol_adj=False)
print("Running 5d/vol-adj signal ...")
pnl_vol = run(use_vol_adj=True)

def stats(p):
    ann = p.mean() * 252
    vol = p.std() * np.sqrt(252)
    sh  = ann / vol if vol > 0 else 0
    cum = (1 + p).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(n=len(p), ann=ann, sharpe=sh, max_dd=dd, cum=cum.iloc[-1]-1, win=(p>0).mean())

s_raw = stats(pnl_raw)
s_vol = stats(pnl_vol)

print(f"\n{'':22}  {'Days':>6}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Max DD':>8}  {'WinRate':>8}  {'Cum Ret':>9}")
print("-" * 80)
for label, s in [("Raw 5d signal (hedged)", s_raw), ("5d/252dvol signal (hedged)", s_vol)]:
    print(f"{label:<22}  {s['n']:>6}  {s['ann']*100:>+8.1f}%  {s['sharpe']:>+8.2f}  "
          f"{s['max_dd']*100:>+8.1f}%  {s['win']*100:>7.0f}%  {s['cum']*100:>+8.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for p, label, color, lw, ls in [
    (pnl_raw, f"Raw 5d  (Sharpe {s_raw['sharpe']:+.2f}, Cum {s_raw['cum']*100:+.1f}%)",
     "#1E88E5", 1.6, "-"),
    (pnl_vol, f"5d/252dVol  (Sharpe {s_vol['sharpe']:+.2f}, Cum {s_vol['cum']*100:+.1f}%)",
     "#E53935", 1.6, "--"),
]:
    cum = (1 + p).cumprod() - 1
    ax.plot(pd.to_datetime(cum.index), cum.values * 100,
            color=color, lw=lw, ls=ls, label=label)

ax.axhline(0, color="black", lw=0.6, ls=":")
ax.set_ylabel("Cumulative return (%)")
ax.set_title(
    "Reversal backtest — SPY-hedged, VIX>20, EW, bottom 25%, daily rebal\n"
    "K100_hierarchical  |  Raw 5d signal vs 5d return / 252d vol",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()

out = _HERE / "results_regime_switch" / "reversal_raw_vs_voladj.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"\nChart -> {out}")
