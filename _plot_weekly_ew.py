#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DAILY_FILE = _HERE.parent.parent / "data" / "daily" / "all.parquet"
OUT        = _HERE / "results_momentum"
LOOKBACK   = 252
REBAL_DAYS = 5
OS_START   = "2026-05-01"

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()

print("Loading prices from parquet ...")
daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
close = close.sort_index().ffill()
avail = [t for t in tickers if t in close.columns]
print(f"  {len(avail)} tickers  {close.index[0]} -> {close.index[-1]}  ({len(close)} days)")
dr    = close[avail].pct_change(fill_method=None)

print("Running weekly EW ret/vol backtest ...")
dates     = close.index.tolist()
positions = pd.Series(0.0, index=avail)
rebal_ctr = 0
pnl_vals, pnl_idx = [], []

for i in range(LOOKBACK + 1, len(dates)):
    day_ret = dr.iloc[i][avail]
    pnl_vals.append(float((positions * day_ret).sum()))
    pnl_idx.append(dates[i])

    if rebal_ctr == 0:
        ret = ((close[avail].iloc[i-1] - close[avail].iloc[i-1-LOOKBACK])
               / close[avail].iloc[i-1-LOOKBACK])
        vol = dr[avail].iloc[i-LOOKBACK:i].std() * np.sqrt(252)
        sig = (ret / vol.replace(0, np.nan)).dropna()
        if len(sig) >= 4:
            ranked = sig.rank(ascending=False)
            n      = len(sig)
            q      = max(1, n // 4)
            longs  = sig[ranked <= q].index.tolist()
            shorts = sig[ranked > n - q].index.tolist()
            positions = pd.Series(0.0, index=avail)
            if longs:  positions[longs]  =  1.0 / len(longs)
            if shorts: positions[shorts] = -1.0 / len(shorts)
    rebal_ctr = (rebal_ctr + 1) % REBAL_DAYS

pnl = pd.Series(pnl_vals, index=pnl_idx)
cum = (1 + pnl).cumprod() - 1

ann_ret = pnl.mean() * 252
ann_vol = pnl.std() * np.sqrt(252)
sharpe  = ann_ret / ann_vol
peak    = (1 + pnl).cumprod().cummax()
max_dd  = (((1 + pnl).cumprod() - peak) / peak).min()
cum_ret = cum.iloc[-1]
print(f"  Sharpe={sharpe:+.2f}  AnnRet={ann_ret*100:+.1f}%  "
      f"MaxDD={max_dd*100:.1f}%  CumRet={cum_ret*100:+.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
is_mask = pd.to_datetime(cum.index) < pd.to_datetime(OS_START)
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(pd.to_datetime(cum.index[is_mask]),  cum.values[is_mask]  * 100,
        color="#1f77b4", lw=1.8, label="IS (backtest)")
ax.plot(pd.to_datetime(cum.index[~is_mask]), cum.values[~is_mask] * 100,
        color="#ff7f0e", lw=2.2, ls="--",
        label=f"OS (live, from {OS_START})")

ax.axvline(pd.to_datetime(OS_START), color="gray", lw=0.9, ls=":")
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_ylabel("Cumulative return (%)")
ax.set_title(
    f"K100_hierarchical  |  252d ret/vol momentum  |  EW  |  weekly rebal\n"
    f"Sharpe {sharpe:+.2f}  |  Ann Ret {ann_ret*100:+.1f}%  |  "
    f"Max DD {max_dd*100:.1f}%  |  Cum Ret {cum_ret*100:+.1f}%",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()
fig.tight_layout()

out = OUT / "weekly_ew_retovol_cumret.png"
OUT.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"Saved -> {out}")
