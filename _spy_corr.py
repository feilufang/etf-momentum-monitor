#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DAILY_FILE = _HERE.parent.parent / "data" / "daily" / "all.parquet"
LOOKBACK   = 252
REBAL_DAYS = 5

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()
daily   = pd.read_parquet(DAILY_FILE, columns=["date","ticker","close"])
daily   = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
close   = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().ffill()
avail   = [t for t in tickers if t in close.columns]
dr      = close[avail].pct_change(fill_method=None)

dates = close.index.tolist()
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
            n = len(sig); q = max(1, n // 4)
            longs  = sig[ranked <= q].index.tolist()
            shorts = sig[ranked > n - q].index.tolist()
            positions = pd.Series(0.0, index=avail)
            if longs:  positions[longs]  =  1.0 / len(longs)
            if shorts: positions[shorts] = -1.0 / len(shorts)
    rebal_ctr = (rebal_ctr + 1) % REBAL_DAYS

pnl = pd.Series(pnl_vals, index=pnl_idx)

spy     = pd.read_csv(_HERE / "spy_daily.csv", index_col=0, parse_dates=True).squeeze()
spy.index = spy.index.strftime("%Y-%m-%d")
spy_ret = spy.pct_change(fill_method=None).reindex(pnl.index).fillna(0)

corr_full = pnl.corr(spy_ret)
beta      = pnl.cov(spy_ret) / spy_ret.var()
alpha_ann = (pnl.mean() - beta * spy_ret.mean()) * 252

print(f"Full sample correlation with SPY: {corr_full:+.3f}")
print(f"Beta to SPY:                      {beta:+.3f}")
print(f"Alpha (ann, Jensen's):            {alpha_ann*100:+.2f}%")
print()
print(f"  {'Year':<6}  {'Corr':>6}  {'Strat':>8}  {'SPY':>8}")
print("  " + "-" * 35)
for yr in range(2017, 2027):
    mask = pd.Series(pnl.index).str.startswith(str(yr)).values
    if mask.sum() < 20:
        continue
    c = pnl[mask].corr(spy_ret[mask])
    s = pnl[mask].mean() * 252 * 100
    p = spy_ret[mask].mean() * 252 * 100
    print(f"  {yr:<6}  {c:>+.3f}  {s:>+7.1f}%  {p:>+7.1f}%")
