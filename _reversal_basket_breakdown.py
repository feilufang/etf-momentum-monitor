#!/usr/bin/env python3
"""
For each active reversal day (VIX[T-1] > 20), classify each name in the
reversal long basket (bottom 25% by 5-day return) as:
  - momentum LONG   (top 25% by 252-day return)
  - momentum SHORT  (bottom 25% by 252-day return)
  - momentum NEUTRAL (middle 50%)

Also shows avg daily P&L contribution by bucket.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DATA_DIR   = _HERE.parent.parent / "data"
DAILY_FILE = DATA_DIR / "daily/all.parquet"
VIX_FILE   = _HERE / "vix_daily.csv"

LOOKBACK    = 252
SIGNAL_DAYS = 5
BOTTOM_PCT  = 0.25
VIX_MIN     = 20

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()

daily = pd.read_parquet(DAILY_FILE, columns=["date","ticker","close"])
daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
close = close.sort_index().ffill()

vix   = pd.read_csv(VIX_FILE, index_col=0, parse_dates=True).squeeze()
vix.index = vix.index.strftime("%Y-%m-%d")
vix_t1 = vix.reindex(close.index).ffill().shift(1)

spy_ret = pd.read_csv(_HERE / "spy_daily.csv", index_col=0, parse_dates=True).squeeze()
spy_ret.index = spy_ret.index.strftime("%Y-%m-%d")
spy_ret = spy_ret.pct_change(fill_method=None).reindex(close.index).fillna(0)

avail     = [t for t in tickers if t in close.columns]
daily_ret = close[avail].pct_change(fill_method=None)
rev_signal = close[avail].pct_change(periods=SIGNAL_DAYS, fill_method=None)

n_select = max(1, int(len(avail) * BOTTOM_PCT))
q        = max(1, len(avail) // 4)

# Per-day breakdown
rows = []

for i in range(LOOKBACK + 1, len(close)):
    date = close.index[i]
    if pd.isna(vix_t1.loc[date]) or vix_t1.loc[date] <= VIX_MIN:
        continue

    # Reversal signal (5-day return as of T-1)
    rev_sig = rev_signal.iloc[i - 1].dropna()
    if len(rev_sig) < n_select:
        continue
    rev_longs = set(rev_sig.nsmallest(n_select).index.tolist())

    # Momentum signal (252-day return as of T-1)
    mom_sig = ((close[avail].iloc[i - 1] - close[avail].iloc[i - 1 - LOOKBACK])
               / close[avail].iloc[i - 1 - LOOKBACK]).dropna()
    if len(mom_sig) < 4:
        continue
    ranked    = mom_sig.rank(ascending=False)
    n_v       = len(mom_sig)
    mom_long  = set(mom_sig[ranked <= q].index.tolist())
    mom_short = set(mom_sig[ranked > n_v - q].index.tolist())

    dr = daily_ret.iloc[i]

    for t in rev_longs:
        if t not in dr.index or pd.isna(dr[t]):
            continue
        if t in mom_long:
            bucket = "mom_long"
        elif t in mom_short:
            bucket = "mom_short"
        else:
            bucket = "neutral"
        spy = spy_ret.loc[date] if date in spy_ret.index else 0.0
        rows.append({
            "date":      date,
            "ticker":    t,
            "bucket":    bucket,
            "ret_raw":   dr[t],
            "ret_hedged": dr[t] - spy,          # long ticker, short equal notional SPY
            "spy_ret":   spy,
            "rev_5d":    rev_sig.get(t, np.nan),
            "mom_252d":  mom_sig.get(t, np.nan),
        })

df = pd.DataFrame(rows)
print(f"Active reversal days (VIX>20): {df['date'].nunique()}")
print(f"Total reversal-long observations: {len(df)}\n")

# ── Breakdown by bucket ────────────────────────────────────────────────────────
totals = len(df)

for mode, col, label_suffix in [("Raw (unhedged)", "ret_raw", ""), ("SPY-hedged", "ret_hedged", " − SPY")]:
    print(f"Breakdown of reversal long basket by momentum bucket  [{mode}]:")
    print(f"  {'Bucket':<14}  {'Obs':>6}  {'% of basket':>12}  {'Avg day ret':>12}  {'Ann ret (est)':>14}")
    print("  " + "-" * 65)
    for bucket, label in [("mom_long","Mom LONG"),("neutral","Neutral"),("mom_short","Mom SHORT")]:
        sub     = df[df["bucket"] == bucket]
        pct     = len(sub) / totals * 100
        avg_ret = sub[col].mean()
        print(f"  {label:<14}  {len(sub):>6}  {pct:>11.1f}%  {avg_ret*100:>+11.3f}%  {avg_ret*252*100:>+13.1f}%")
    print()

# ── Daily P&L contribution by bucket (hedged) ────────────────────────────────
print("Avg P&L contribution per active day — SPY-hedged (equal-weight basket of 25):")
daily_by_bucket = (df.groupby(["date","bucket"])["ret_hedged"].mean().unstack(fill_value=0))
daily_counts    = (df.groupby(["date","bucket"])["ret_hedged"].count().unstack(fill_value=0))
n_total_per_day = daily_counts.sum(axis=1)

for col in ["mom_long", "neutral", "mom_short"]:
    if col not in daily_counts.columns:    daily_counts[col]    = 0
    if col not in daily_by_bucket.columns: daily_by_bucket[col] = 0

weighted_contrib = pd.DataFrame({
    col: (daily_by_bucket[col] * daily_counts[col] / n_total_per_day)
    for col in ["mom_long", "neutral", "mom_short"]
})

print(f"  {'Bucket':<14}  {'Avg count/day':>14}  {'Avg contribution/day':>22}  {'Sharpe':>8}")
print("  " + "-" * 65)
for bucket, label in [("mom_long","Mom LONG"),("neutral","Neutral"),("mom_short","Mom SHORT")]:
    avg_count = daily_counts[bucket].mean()
    contrib   = weighted_contrib[bucket]
    avg_c     = contrib.mean()
    sharpe_c  = (avg_c / contrib.std() * np.sqrt(252)) if contrib.std() > 0 else 0
    print(f"  {label:<14}  {avg_count:>14.1f}  {avg_c*100:>+21.3f}%  {sharpe_c:>+8.2f}")

print()

# ── Time series: what fraction of basket is mom_short over time ───────────────
frac_short = (daily_counts.get("mom_short", pd.Series(0, index=daily_counts.index))
              / n_total_per_day * 100)
print(f"Fraction of reversal basket that is also in momentum SHORT book:")
print(f"  Overall mean : {frac_short.mean():.1f}%")
print(f"  Overall median: {frac_short.median():.1f}%")
print(f"  Max on any day: {frac_short.max():.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(_HERE / "results_regime_switch" / "reversal_basket_breakdown.csv", index=False)
print(f"\nDetailed CSV -> results_regime_switch/reversal_basket_breakdown.csv")
