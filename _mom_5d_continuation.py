#!/usr/bin/env python3
"""
For each historical date, find names in BOTH:
  - 252d ret/vol top quartile  (mom long book)
  - 5d ret/daily_std top quartile  (recent 5d winner)

Measure their equal-weight average forward returns at 3, 5, 10, 20 days.

Same analysis on the SHORT side:
  - 252d ret/vol bottom quartile  (mom short book)
  - 5d ret/daily_std bottom quartile  (recent 5d loser)
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
MOM_LB     = 252
VOL_LB     = 252
SIG5_LB    = 252   # lookback for daily std in 5d signal
FWDS       = [3, 5, 10, 20]

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()
print(f"Universe: {len(tickers)} tickers")

daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
close = (
    daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    .sort_index()
    .ffill()
)
avail = [t for t in tickers if t in close.columns]
dr    = close[avail].pct_change(fill_method=None)
print(f"History: {close.index[0]} -> {close.index[-1]}  ({len(close)} days, {len(avail)} tickers)")

# ── collect per-date forward returns ─────────────────────────────────────────

records_long  = []   # (date, ticker, fwd_3, fwd_5, fwd_10, fwd_20)
records_short = []

warmup = MOM_LB + 1
n_dates = len(close)
max_fwd = max(FWDS)

for i in range(warmup, n_dates - max_fwd):
    date_i = close.index[i]

    # 252d ret/vol signal (use prices up to i-1, signal acts on close[i-1])
    px_mom = close[avail].iloc[i - MOM_LB - 1 : i]   # MOM_LB+1 rows → MOM_LB returns
    ret252 = (px_mom.iloc[-1] - px_mom.iloc[0]) / px_mom.iloc[0]
    vol252 = dr[avail].iloc[i - MOM_LB : i].std() * np.sqrt(252)
    sig252 = (ret252 / vol252.replace(0, np.nan)).dropna()

    q      = max(1, len(sig252) // 4)
    ranked = sig252.rank(ascending=False)
    n_v    = len(ranked)
    mom_long_set  = set(ranked[ranked <= q].index)
    mom_short_set = set(ranked[ranked > n_v - q].index)

    # 5d ret / 252d daily_std signal (same lookback for std)
    if i < SIG5_LB + 5:
        continue
    ret5   = (close[avail].iloc[i - 1] - close[avail].iloc[i - 6]) / close[avail].iloc[i - 6]
    std252 = dr[avail].iloc[i - SIG5_LB : i].std()
    sig5   = (ret5 / std252.replace(0, np.nan)).dropna()

    q5     = max(1, len(sig5) // 4)
    ranked5 = sig5.rank(ascending=False)
    n5     = len(ranked5)
    top5_set    = set(ranked5[ranked5 <= q5].index)
    bottom5_set = set(ranked5[ranked5 > n5 - q5].index)

    # intersections
    continuation_long  = mom_long_set  & top5_set      # mom long + recent 5d winner
    continuation_short = mom_short_set & bottom5_set   # mom short + recent 5d loser

    # forward returns: close[i + fwd] / close[i] - 1
    for ticker in continuation_long:
        if ticker not in close.columns:
            continue
        row = {"date": date_i, "ticker": ticker}
        for f in FWDS:
            row[f"fwd_{f}d"] = (
                close[ticker].iloc[i + f] / close[ticker].iloc[i] - 1
            )
        records_long.append(row)

    for ticker in continuation_short:
        if ticker not in close.columns:
            continue
        row = {"date": date_i, "ticker": ticker}
        for f in FWDS:
            row[f"fwd_{f}d"] = (
                close[ticker].iloc[i + f] / close[ticker].iloc[i] - 1
            )
        records_short.append(row)

df_long  = pd.DataFrame(records_long)
df_short = pd.DataFrame(records_short)
print(f"\nObservations — long continuation: {len(df_long)}, short continuation: {len(df_short)}")

# ── summary stats ────────────────────────────────────────────────────────────

fwd_cols = [f"fwd_{f}d" for f in FWDS]

def summary(df, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {len(df)} obs  ({df['ticker'].nunique()} unique tickers, "
          f"{df['date'].nunique()} unique dates)")
    print(f"{'='*70}")
    hdr = f"{'Horizon':<12}  {'Mean Ret':>10}  {'Median':>10}  {'Win Rate':>10}  "
    hdr += f"{'Ann Equiv':>10}  {'t-stat':>8}"
    print(hdr)
    print("-" * 70)
    for f, col in zip(FWDS, fwd_cols):
        vals = df[col].dropna()
        mean  = vals.mean()
        med   = vals.median()
        win   = (vals > 0).mean()
        ann   = mean * (252 / f)
        tstat = mean / (vals.std() / np.sqrt(len(vals))) if vals.std() > 0 else np.nan
        print(f"{f:>2}d            {mean*100:>+9.2f}%  {med*100:>+9.2f}%  "
              f"{win*100:>9.1f}%  {ann*100:>+9.1f}%  {tstat:>8.2f}")

summary(df_long,  "252d Mom LONG  +  5d top quartile  → forward returns")
summary(df_short, "252d Mom SHORT +  5d bottom quartile  → forward returns")

# ── SPY benchmark (same dates, same horizons) ────────────────────────────────

if "SPY" in close.columns:
    spy_dates_long  = df_long["date"].unique()
    spy_dates_short = df_short["date"].unique()

    def spy_fwd(dates):
        rows = []
        for d in dates:
            i = close.index.get_loc(d)
            if i + max_fwd >= len(close):
                continue
            row = {}
            for f in FWDS:
                row[f"fwd_{f}d"] = close["SPY"].iloc[i + f] / close["SPY"].iloc[i] - 1
            rows.append(row)
        return pd.DataFrame(rows)

    spy_l = spy_fwd(spy_dates_long)
    spy_s = spy_fwd(spy_dates_short)

    print(f"\n{'='*70}")
    print("  SPY benchmark on same dates (long-continuation days)")
    print(f"{'='*70}")
    for f, col in zip(FWDS, fwd_cols):
        if col not in spy_l.columns: continue
        vals = spy_l[col].dropna()
        print(f"  {f:>2}d  mean {vals.mean()*100:>+.2f}%")

    print(f"\n  SPY benchmark on same dates (short-continuation days)")
    for f, col in zip(FWDS, fwd_cols):
        if col not in spy_s.columns: continue
        vals = spy_s[col].dropna()
        print(f"  {f:>2}d  mean {vals.mean()*100:>+.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, df, label, color in [
    (axes[0], df_long,  "Mom LONG + 5d Top Quartile", "#1E88E5"),
    (axes[1], df_short, "Mom SHORT + 5d Bottom Quartile", "#E53935"),
]:
    means  = [df[f"fwd_{f}d"].mean() * 100 for f in FWDS]
    stderrs = [df[f"fwd_{f}d"].std() / np.sqrt(len(df[f"fwd_{f}d"].dropna())) * 100
               for f in FWDS]
    ax.bar([str(f) + "d" for f in FWDS], means, color=color, alpha=0.75)
    ax.errorbar([str(f) + "d" for f in FWDS], means, yerr=[1.96 * se for se in stderrs],
                fmt="none", color="black", capsize=4, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Mean forward return (%)")
    ax.set_xlabel("Horizon")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "K100_hierarchical  |  Forward returns: momentum + 5d signal overlap\n"
    "Error bars = 95% CI  |  All dates 2016–2026",
    fontsize=10,
)
fig.tight_layout()
out = _HERE / "results_momentum" / "mom_5d_continuation_fwdrets.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"\nChart -> {out}")
