#!/usr/bin/env python3
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
OUT        = _HERE / "results_momentum"
LOOKBACK   = 252
REBAL_DAYS = 5

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()
daily   = pd.read_parquet(DAILY_FILE, columns=["date","ticker","close"])
daily   = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
close   = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().ffill()
avail   = [t for t in tickers if t in close.columns]
dr      = close[avail].pct_change(fill_method=None)

# VIX (T-1 to avoid lookahead)
vix = pd.read_csv(_HERE / "vix_daily.csv", index_col=0, parse_dates=True).squeeze()
vix.index = vix.index.strftime("%Y-%m-%d")
vix_t1 = vix.reindex(close.index).ffill().shift(1)

print("Running weekly EW ret/vol backtest ...")
dates = close.index.tolist()
positions = pd.Series(0.0, index=avail)
rebal_ctr = 0
rows = []

for i in range(LOOKBACK + 1, len(dates)):
    date    = dates[i]
    day_ret = dr.iloc[i][avail]
    pnl     = float((positions * day_ret).sum())
    vix_val = vix_t1.loc[date] if date in vix_t1.index else np.nan
    rows.append({"date": date, "pnl": pnl, "vix": vix_val})

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

df = pd.DataFrame(rows).set_index("date")

# VIX buckets
bins   = [0, 12, 15, 20, 25, 30, 999]
labels = ["<12", "12-15", "15-20", "20-25", "25-30", ">30"]
df["vix_bucket"] = pd.cut(df["vix"], bins=bins, labels=labels)

# ── Stats table ───────────────────────────────────────────────────────────────
print(f"\nPerformance conditioned on VIX[T-1]:")
print(f"  {'VIX':>8}  {'Days':>6}  {'Ann Ret':>9}  {'Sharpe':>8}  {'WinRate':>8}  {'Max DD':>8}  {'Corr SPY':>9}")
print("  " + "-" * 70)

spy     = pd.read_csv(_HERE / "spy_daily.csv", index_col=0, parse_dates=True).squeeze()
spy.index = spy.index.strftime("%Y-%m-%d")
spy_ret = spy.pct_change(fill_method=None).reindex(df.index).fillna(0)

bucket_pnls = {}
for b in labels:
    sub = df[df["vix_bucket"] == b]["pnl"].dropna()
    if len(sub) < 5:
        continue
    ann_ret  = sub.mean() * 252
    ann_vol  = sub.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    win_rate = (sub > 0).mean()
    cum      = (1 + sub).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()
    corr_spy = sub.corr(spy_ret.reindex(sub.index).fillna(0))
    bucket_pnls[b] = sub
    print(f"  {b:>8}  {len(sub):>6}  {ann_ret*100:>+8.1f}%  {sharpe:>+8.2f}  "
          f"{win_rate*100:>7.0f}%  {max_dd*100:>+8.1f}%  {corr_spy:>+9.3f}")

# ── Chart: cumulative return by VIX bucket ────────────────────────────────────
colors = {"<12": "#1565C0", "12-15": "#42A5F5", "15-20": "#FFA726",
          "20-25": "#EF5350", "25-30": "#B71C1C", ">30": "#4A148C"}

fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                          gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12},
                          sharex=False)

# Top: cumulative return per VIX bucket
ax1 = axes[0]
for b in labels:
    if b not in bucket_pnls:
        continue
    sub = bucket_pnls[b]
    cum = (1 + sub).cumprod() - 1
    ann = sub.mean() * 252
    sh  = ann / (sub.std() * np.sqrt(252)) if sub.std() > 0 else 0
    ax1.plot(range(len(cum)), cum.values * 100,
             color=colors[b], linewidth=1.6,
             label=f"VIX {b}  ({len(sub)}d  Sharpe {sh:+.2f}  Ann {ann*100:+.1f}%)")

ax1.axhline(0, color="black", lw=0.6, ls="--")
ax1.set_ylabel("Cumulative return (%)")
ax1.set_title("K100 weekly EW ret/vol momentum — performance by VIX[T-1] regime", fontsize=11)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(alpha=0.25)
ax1.set_xlabel("Trading days in regime")

# Bottom: bar chart of annualised return per bucket
ax2 = axes[1]
b_list  = [b for b in labels if b in bucket_pnls]
ann_rets = [bucket_pnls[b].mean() * 252 * 100 for b in b_list]
bar_cols = [colors[b] for b in b_list]
bars = ax2.bar(b_list, ann_rets, color=bar_cols, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, ann_rets):
    ax2.text(bar.get_x() + bar.get_width()/2, v + (0.3 if v >= 0 else -0.8),
             f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=9, fontweight="bold")
ax2.axhline(0, color="black", lw=0.7)
ax2.set_ylabel("Ann return (%)")
ax2.set_xlabel("VIX[T-1] bucket")
ax2.set_title("Annualised return by VIX regime", fontsize=10)
ax2.grid(axis="y", alpha=0.25)
ax2.spines[["top","right"]].set_visible(False)

fig.tight_layout()
out = OUT / "mom_vix_conditioned.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out}")
