#!/usr/bin/env python3
"""
Backtest two long-only baskets, daily rebal, EW:
  Leg A: 5d top-quartile  ∩  252d mom LONG   (momentum continuation)
  Leg B: 5d bottom-quartile ∩  252d mom SHORT  (reversal of the short book)
  Combined: equal-weight A + B

Signal definitions (no look-ahead):
  - 252d ret/vol and 5d ret/daily_std computed from prices known at close[i-1]
  - Trade executed at close[i]  (T+1 execution)
"""
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
MOM_LB     = 252

tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()
print(f"Universe: {len(tickers)} tickers")

daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
daily = daily[daily["ticker"].isin(tickers + ["SPY"]) & daily["close"].notna() & daily["close"].gt(0)]
close = (
    daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    .sort_index()
    .ffill()
)
avail  = [t for t in tickers if t in close.columns]
dr     = close[avail].pct_change(fill_method=None)
spy_dr = close["SPY"].pct_change(fill_method=None) if "SPY" in close.columns else pd.Series(0.0, index=close.index)
print(f"History: {close.index[0]} -> {close.index[-1]}  ({len(close)} days, {len(avail)} tickers)")

warmup = MOM_LB + 5 + 1

pnl_a, pnl_b, pnl_combo = [], [], []
idx = []

for i in range(warmup, len(close)):
    date_i = close.index[i]

    # 252d ret/vol
    ret252 = (close[avail].iloc[i-1] - close[avail].iloc[i-1-MOM_LB]) / close[avail].iloc[i-1-MOM_LB]
    vol252 = dr[avail].iloc[i-MOM_LB:i].std() * np.sqrt(252)
    sig252 = (ret252 / vol252.replace(0, np.nan)).dropna()

    q     = max(1, len(sig252) // 4)
    rk252 = sig252.rank(ascending=False)
    n_v   = len(rk252)
    mom_long_set  = set(rk252[rk252 <= q].index)
    mom_short_set = set(rk252[rk252 > n_v - q].index)

    # 5d ret / 252d daily_std
    ret5   = (close[avail].iloc[i-1] - close[avail].iloc[i-6]) / close[avail].iloc[i-6]
    std252 = dr[avail].iloc[i-MOM_LB:i].std()
    sig5   = (ret5 / std252.replace(0, np.nan)).dropna()

    q5   = max(1, len(sig5) // 4)
    rk5  = sig5.rank(ascending=False)
    n5   = len(rk5)
    top5_set    = set(rk5[rk5 <= q5].index)
    bottom5_set = set(rk5[rk5 > n5 - q5].index)

    leg_a = list(mom_long_set  & top5_set)
    leg_b = list(mom_short_set & bottom5_set)

    if not leg_a and not leg_b:
        continue

    ret_i = dr[avail].iloc[i]

    def ew_ret(names):
        vals = ret_i[names].dropna()
        return vals.mean() if len(vals) > 0 else np.nan

    ra = ew_ret(leg_a)
    rb = ew_ret(leg_b)
    legs_avail = [r for r in [ra, rb] if not pd.isna(r)]
    rc = np.mean(legs_avail) if legs_avail else np.nan

    pnl_a.append(ra)
    pnl_b.append(rb)
    pnl_combo.append(rc)
    idx.append(date_i)

pnl_a     = pd.Series(pnl_a,     index=idx).dropna()
pnl_b     = pd.Series(pnl_b,     index=idx).dropna()
pnl_combo = pd.Series(pnl_combo, index=idx).dropna()

# ── stats ─────────────────────────────────────────────────────────────────────

def stats(p, label, spy):
    ann  = p.mean() * 252
    vol  = p.std() * np.sqrt(252)
    sh   = ann / vol if vol > 0 else 0
    cum  = (1 + p).cumprod()
    dd   = ((cum - cum.cummax()) / cum.cummax()).min()
    win  = (p > 0).mean()
    cumr = cum.iloc[-1] - 1

    # beta / alpha vs SPY
    spy_a = spy.reindex(p.index).dropna()
    p_a   = p.reindex(spy_a.index)
    beta  = np.cov(p_a, spy_a)[0, 1] / np.var(spy_a) if len(spy_a) > 10 else np.nan
    alpha = (p_a.mean() - beta * spy_a.mean()) * 252 if not np.isnan(beta) else np.nan

    # hedged stats
    ph   = p_a - beta * spy_a
    sh_h = ph.mean() * 252 / (ph.std() * np.sqrt(252)) if ph.std() > 0 else 0

    print(f"\n  {label}")
    print(f"    N={len(p):>5}  Ann={ann*100:>+6.1f}%  Sharpe={sh:>+5.2f}  "
          f"MaxDD={dd*100:>+6.1f}%  Win={win*100:.0f}%  Cum={cumr*100:>+7.1f}%")
    print(f"    Beta={beta:>+5.2f}  Alpha(ann)={alpha*100:>+5.1f}%  "
          f"Beta-hedged Sharpe={sh_h:>+5.2f}")
    return dict(ann=ann, sharpe=sh, max_dd=dd, cum=cumr, win=win, beta=beta, alpha=alpha)

spy_full  = spy_dr.reindex(pnl_combo.index).dropna()
spy_ann   = spy_full.mean() * 252
spy_vol   = spy_full.std() * np.sqrt(252)
spy_sh    = spy_ann / spy_vol
spy_cum   = (1 + spy_full).cumprod().iloc[-1] - 1
spy_dd    = ((( 1 + spy_full).cumprod() / (1 + spy_full).cumprod().cummax()) - 1).min()

print(f"\nSPY benchmark (same dates): Ann={spy_ann*100:>+6.1f}%  Sharpe={spy_sh:>+5.2f}  "
      f"MaxDD={spy_dd*100:>+6.1f}%  Cum={spy_cum*100:>+7.1f}%")
print(f"\n{'='*80}")
print("Strategy results (absolute / unhedged):")
print("="*80)
s_a     = stats(pnl_a,     "Leg A: 5d best  ∩ mom long", spy_dr)
s_b     = stats(pnl_b,     "Leg B: 5d worst ∩ mom short", spy_dr)
s_combo = stats(pnl_combo, "Combined (A+B EW)", spy_dr)

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# left: absolute cumulative returns
ax = axes[0]
spy_cum_plot = (1 + spy_dr.reindex(pnl_combo.index).fillna(0)).cumprod() - 1
for p, label, color, lw, ls in [
    (pnl_a,     f"Leg A: 5d best ∩ mom long\nSh={s_a['sharpe']:+.2f}  β={s_a['beta']:+.2f}  α={s_a['alpha']*100:+.1f}%",
     "#1E88E5", 1.4, "-"),
    (pnl_b,     f"Leg B: 5d worst ∩ mom short\nSh={s_b['sharpe']:+.2f}  β={s_b['beta']:+.2f}  α={s_b['alpha']*100:+.1f}%",
     "#43A047", 1.4, "-"),
    (pnl_combo, f"Combined EW\nSh={s_combo['sharpe']:+.2f}  β={s_combo['beta']:+.2f}  α={s_combo['alpha']*100:+.1f}%",
     "#E53935", 2.0, "--"),
]:
    cum = (1 + p).cumprod() - 1
    ax.plot(pd.to_datetime(cum.index), cum.values * 100, color=color, lw=lw, ls=ls, label=label)
ax.plot(pd.to_datetime(spy_cum_plot.index), spy_cum_plot.values * 100,
        color="black", lw=1.2, ls=":", alpha=0.6, label="SPY")
ax.axhline(0, color="gray", lw=0.6)
ax.set_ylabel("Cumulative return (%)")
ax.set_title("Absolute returns", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# right: beta-hedged cumulative returns
ax = axes[1]
for p, label, color, lw, ls, s in [
    (pnl_a,     "Leg A (beta-hedged)", "#1E88E5", 1.4, "-",  s_a),
    (pnl_b,     "Leg B (beta-hedged)", "#43A047", 1.4, "-",  s_b),
    (pnl_combo, "Combined (beta-hedged)", "#E53935", 2.0, "--", s_combo),
]:
    spy_a = spy_dr.reindex(p.index).fillna(0)
    ph    = p - s["beta"] * spy_a
    cum_h = (1 + ph).cumprod() - 1
    ax.plot(pd.to_datetime(cum_h.index), cum_h.values * 100, color=color, lw=lw, ls=ls, label=label)
ax.axhline(0, color="gray", lw=0.6)
ax.set_ylabel("Cumulative return (%)")
ax.set_title("Beta-hedged (alpha) returns", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle(
    "K100_hierarchical  |  Long (5d best ∩ mom long) + Long (5d worst ∩ mom short)  |  EW daily rebal\n"
    "Left: absolute  |  Right: beta-hedged",
    fontsize=10,
)
fig.tight_layout()
out = _HERE / "results_momentum" / "mom_5d_combo_backtest.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"\nChart -> {out}")
