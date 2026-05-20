#!/usr/bin/env python3
"""
GitHub Actions version of the daily momentum monitor.
Fetches prices from Yahoo Finance — no local data files required.

Environment variables (set as GitHub Secrets):
    GMAIL_USER           your Gmail address
    GMAIL_APP_PASSWORD   16-char app password from myaccount.google.com/apppasswords

Usage (local test):
    python monitor_gha.py --preview
    python monitor_gha.py --to rogerwugang@gmail.com
"""

import io
import os
import smtplib
import sys
from datetime import date
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent
UNIVERSE_CSV = _HERE / "results_corr" / "selected_K100_hierarchical.csv"
NAMES_CSV    = _HERE / "etf_names_K100_hierarchical.csv"

UNIVERSE    = "K100_hierarchical"
LOOKBACK    = 252
SKIP        = 0
REBAL_DAYS  = 1       # daily rebalance
COV_LB      = 63      # covariance lookback for ERC
SIGNAL_DAYS = 5       # short-term reversal window
ADV_DAYS    = 21      # trailing days for ADV
OS_START   = "2026-05-01"
DATA_START = "2016-01-01"   # full history for backtest chart
RECIPIENTS = ["feilu.fang@gmail.com"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    return pd.read_csv(UNIVERSE_CSV)["ticker"].tolist()


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Extract a single field (e.g. Close, Volume) from a yfinance MultiIndex result."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()
        if field in lvl0:
            return raw[field].copy()
        elif field in lvl1:
            return raw.xs(field, axis=1, level=1).copy()
        else:
            raise RuntimeError(f"Cannot find '{field}' in MultiIndex columns: {raw.columns[:10].tolist()}")
    else:
        col = raw[[field]].copy()
        col.columns = tickers[:1]
        return col


def load_prices_yf(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (close, volume) DataFrames indexed by YYYY-MM-DD strings."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    print(f"  Fetching Yahoo Finance data for {len(tickers)} tickers from {DATA_START} ...")
    raw = yf.download(
        tickers,
        start=DATA_START,
        auto_adjust=True,
        progress=False,
    )
    print(f"  Raw shape: {raw.shape}  columns type: {type(raw.columns).__name__}")
    if isinstance(raw.columns, pd.MultiIndex):
        print(f"  MultiIndex level-0 samples: {raw.columns.get_level_values(0).unique().tolist()[:5]}")

    close  = _extract_field(raw, "Close",  tickers)
    volume = _extract_field(raw, "Volume", tickers)

    for df in (close, volume):
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        df.sort_index(inplace=True)

    close  = close.ffill().dropna(axis=1, how="all")
    volume = volume.fillna(0)

    available = [t for t in tickers if t in close.columns]
    missing   = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"  Warning: {len(missing)} tickers not on Yahoo Finance: {missing}")
    print(f"  {len(close):,} days  {close.index[0]} -> {close.index[-1]}"
          f"  ({len(available)} tickers available)")
    return close, volume


def load_vix_yf() -> pd.Series:
    """Return ^VIX daily closes as a Series indexed by YYYY-MM-DD strings."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    print("  Fetching ^VIX ...")
    raw = yf.download("^VIX", start=DATA_START, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].iloc[:, 0]
    else:
        vix = raw["Close"]
    vix.index = pd.to_datetime(vix.index).strftime("%Y-%m-%d")
    return vix.sort_index().ffill()


def load_spy_yf() -> pd.Series:
    """Return SPY daily close prices as a Series indexed by YYYY-MM-DD strings."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    print("  Fetching SPY ...")
    raw = yf.download("SPY", start=DATA_START, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        spy = raw["Close"].iloc[:, 0]
    else:
        spy = raw["Close"]
    spy.index = pd.to_datetime(spy.index).strftime("%Y-%m-%d")
    return spy.sort_index().ffill()


def compute_adv(close: pd.DataFrame, volume: pd.DataFrame, n: int = ADV_DAYS) -> pd.Series:
    """Trailing n-day average daily dollar volume for each ticker (latest row)."""
    dv = close * volume
    return dv.iloc[-n:].mean()


def load_etf_names() -> dict[str, str]:
    if NAMES_CSV.exists():
        df = pd.read_csv(NAMES_CSV)
        return df.set_index("ticker")["name"].to_dict()
    return {}


# ── ERC helpers ───────────────────────────────────────────────────────────────

def _erc_weights(cov: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize
    n  = cov.shape[0]
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


def _erc_book(members: list[str], dr: pd.DataFrame, i: int) -> np.ndarray:
    if len(members) == 1:
        return np.array([1.0])
    hist = dr[members].iloc[max(0, i - COV_LB):i].dropna()
    if len(hist) < 10:
        return np.ones(len(members)) / len(members)
    cov = hist.cov().values.copy()
    cov += np.eye(len(members)) * np.diag(cov).mean() * 0.10
    return _erc_weights(cov)


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    tickers: list[str],
    lookback: int,
    rebal_days: int,
    skip: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Returns (daily_pnl, current_positions). Ranks by ret/vol; ERC weights."""
    px        = close[tickers].dropna(how="all")
    dr        = px.pct_change(fill_method=None)
    dates     = px.index.tolist()
    if len(dates) < lookback + skip + rebal_days + 1:
        return pd.Series(dtype=float), pd.Series(0.0, index=tickers)

    daily_pnl     = []
    positions     = pd.Series(0.0, index=tickers)
    rebal_counter = 0

    for i in range(lookback + skip + 1, len(dates)):
        today   = dates[i]
        day_ret = dr.iloc[i][tickers]
        day_pnl = (positions * day_ret).sum()
        daily_pnl.append((today, float(day_pnl)))

        if rebal_counter == 0:
            signal_px = px.iloc[i - 1 - skip][tickers]
            past_px   = px.iloc[i - 1 - skip - lookback][tickers]
            ret       = (signal_px - past_px) / past_px
            vol       = dr[tickers].iloc[i - lookback:i].std() * np.sqrt(252)
            signal    = (ret / vol.replace(0, np.nan)).dropna()
            if len(signal) >= 4:
                ranked  = signal.rank(ascending=False)
                n_valid = len(signal)
                q       = max(1, n_valid // 4)
                longs   = signal[ranked <= q].index.tolist()
                shorts  = signal[ranked > n_valid - q].index.tolist()
                positions = pd.Series(0.0, index=tickers)
                wl = _erc_book(longs,  dr, i)
                ws = _erc_book(shorts, dr, i)
                for t, w in zip(longs,  wl): positions[t] =  w
                for t, w in zip(shorts, ws): positions[t] = -w
            else:
                positions = pd.Series(0.0, index=tickers)

        rebal_counter = (rebal_counter + 1) % rebal_days

    pnl = pd.DataFrame(daily_pnl, columns=["date", "pnl"]).set_index("date")["pnl"]
    return pnl, positions


def _ret_over_vol(px: pd.DataFrame, window: int) -> pd.Series:
    """Return / annualised vol over the last `window` days (Sharpe-like ratio)."""
    if len(px) < window + 1:
        return pd.Series(dtype=float)
    ret    = ((px.iloc[-1] - px.iloc[-1 - window]) / px.iloc[-1 - window]).dropna()
    dr     = px.pct_change(fill_method=None).iloc[-window:]
    vol    = dr.std() * np.sqrt(252)
    ratio  = ret / vol.replace(0, np.nan)
    return ratio.dropna()


def get_current_signal(
    close: pd.DataFrame, tickers: list[str], lookback: int, skip: int
) -> pd.Series:
    px = close[tickers].dropna(how="all")
    if skip > 0:
        px = px.iloc[:-skip]
    return _ret_over_vol(px, lookback)


def get_100d_signal(close: pd.DataFrame, tickers: list[str]) -> pd.Series:
    px = close[tickers].dropna(how="all")
    return _ret_over_vol(px, 100)


def get_raw_return(close: pd.DataFrame, tickers: list[str], window: int) -> pd.Series:
    px = close[tickers].dropna(how="all")
    if len(px) < window + 1:
        return pd.Series(dtype=float)
    return ((px.iloc[-1] - px.iloc[-1 - window]) / px.iloc[-1 - window]).dropna()


def get_5d_signal(close: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """5-day return / 252-day daily std  (how many daily σ moved in 5 days)."""
    px = close[tickers].dropna(how="all")
    if len(px) < 253:
        return pd.Series(dtype=float)
    ret5      = ((px.iloc[-1] - px.iloc[-6]) / px.iloc[-6]).dropna()
    daily_std = px.pct_change(fill_method=None).iloc[-252:].std()
    return (ret5 / daily_std.reindex(ret5.index).replace(0, np.nan)).dropna()


# ── Statistics ─────────────────────────────────────────────────────────────────

def stats(pnl: pd.Series) -> dict:
    if pnl.empty:
        return {"ann_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0, "cum_ret": 0.0, "n_days": 0}
    cum  = (1 + pnl).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    ann  = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    return {
        "ann_ret": float(ann),
        "sharpe":  float(ann / vol) if vol > 0 else 0.0,
        "max_dd":  float(dd.min()),
        "cum_ret": float(cum.iloc[-1] - 1),
        "n_days":  int(len(pnl)),
    }


# ── Chart ───────────────────────────────────────────────────────────────────────

def make_chart(pnl: pd.Series, os_start: str) -> bytes:
    pnl_clean = pnl.dropna()
    cum       = (1 + pnl_clean).cumprod() - 1
    dd        = ((1 + pnl_clean).cumprod() / (1 + pnl_clean).cumprod().cummax() - 1)
    is_mask   = pd.to_datetime(cum.index) < pd.to_datetime(os_start)
    dates_dt  = pd.to_datetime(cum.index)

    # summary stats
    ann_ret = pnl_clean.mean() * 252
    ann_vol = pnl_clean.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd  = dd.min()
    cum_ret = cum.iloc[-1]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    ax1.plot(dates_dt[is_mask],  cum.values[is_mask]  * 100,
             color="#1f77b4", lw=1.6, label="IS (backtest)")
    if (~is_mask).any():
        ax1.plot(dates_dt[~is_mask], cum.values[~is_mask] * 100,
                 color="#ff7f0e", lw=2.2, ls="--",
                 label=f"OS (live, from {os_start})")

    ax1.axvline(pd.to_datetime(os_start), color="gray", lw=0.9, ls=":")
    ax1.axhline(0, color="black", lw=0.6, ls="--")
    ax1.set_ylabel("Cumulative return (%)")
    ax1.set_title(
        f"{UNIVERSE}  |  252d ret/vol momentum  |  EW  |  weekly rebal  |  T+1 execution\n"
        f"Sharpe {sharpe:+.2f}  |  Ann Ret {ann_ret*100:+.1f}%  |  "
        f"Max DD {max_dd*100:.1f}%  |  Cum Ret {cum_ret*100:+.1f}%",
        fontsize=10,
    )
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.fill_between(dates_dt, dd.values * 100, 0,
                     color="#E53935", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(min(dd.min() * 100 * 1.2, -1), 1)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_reversal_backtest(
    close: pd.DataFrame,
    tickers: list[str],
    spy_ret: pd.Series,
    bottom_pct: float = 0.25,
    vol_lb: int = 252,
) -> pd.Series:
    """5-day reversal: long bottom-25% by 5d/daily_std, all days, EW, SPY-hedged."""
    px       = close[tickers]
    dr       = px.pct_change(fill_method=None)
    n_select = max(1, int(len(tickers) * bottom_pct))
    rev_raw  = px.pct_change(periods=5, fill_method=None)

    pnl_vals, pnl_idx = [], []
    for i in range(vol_lb + 1, len(close)):
        date_i = close.index[i]

        sig_5d = rev_raw.iloc[i - 1].dropna()
        if len(sig_5d) < n_select:
            continue

        daily_std = dr.iloc[i - vol_lb:i].std()
        signal    = (sig_5d / daily_std.reindex(sig_5d.index).replace(0, np.nan)).dropna()
        if len(signal) < n_select:
            continue

        selected = signal.nsmallest(n_select).index.tolist()
        day_rets = dr.iloc[i][selected].dropna()
        if day_rets.empty:
            continue

        spy_d = float(spy_ret.loc[date_i]) if date_i in spy_ret.index else 0.0
        pnl_vals.append(day_rets.mean() - spy_d)
        pnl_idx.append(date_i)

    return pd.Series(pnl_vals, index=pnl_idx)


def make_reversal_chart(pnl: pd.Series) -> bytes:
    pnl_clean = pnl.dropna()
    cum      = (1 + pnl_clean).cumprod() - 1
    dd       = (1 + pnl_clean).cumprod() / (1 + pnl_clean).cumprod().cummax() - 1
    dates_dt = pd.to_datetime(pnl_clean.index)

    ann_ret = pnl_clean.mean() * 252
    ann_vol = pnl_clean.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd  = dd.min()
    cum_ret = cum.iloc[-1]
    n_days  = len(pnl_clean)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    ax1.plot(dates_dt, cum.values * 100, color="#E53935", lw=1.6)
    ax1.axhline(0, color="black", lw=0.6, ls="--")
    ax1.set_ylabel("Cumulative return (%)")
    ax1.set_title(
        f"{UNIVERSE}  |  5-day reversal  |  all days  |  EW  |  SPY-hedged\n"
        f"Signal: 5d return / 252d daily σ  |  Bottom 25%  |  "
        f"Sharpe {sharpe:+.2f}  |  Cum Ret {cum_ret*100:+.1f}%  |  "
        f"Max DD {max_dd*100:.1f}%  |  N={n_days} days",
        fontsize=10,
    )
    ax1.grid(alpha=0.25)

    ax2.fill_between(dates_dt, dd.values * 100, 0, color="#E53935", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(min(dd.min() * 100 * 1.2, -1), 1)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── HTML ───────────────────────────────────────────────────────────────────────

def _stats_row(s: dict, footnote: bool = False) -> str:
    star = " *" if footnote else ""
    return f"""
<table cellpadding="8" cellspacing="0"
       style="border-collapse:collapse;width:100%;margin-bottom:6px;">
<tr style="background:#f0f0f0;font-weight:bold">
  <th>Sharpe{star}</th><th>Ann Return</th>
  <th>Cum Return</th><th>Max DD</th><th>Days</th>
</tr>
<tr style="text-align:center">
  <td>{s['sharpe']:+.2f}</td>
  <td>{s['ann_ret']:+.1%}</td>
  <td>{s['cum_ret']:+.1%}</td>
  <td>{s['max_dd']:.1%}</td>
  <td>{s['n_days']}</td>
</tr>
</table>
{"<p style='font-size:11px;color:#888'>* Sharpe not meaningful with fewer than 20 trading days</p>" if footnote else ""}"""


def _momentum_table(label: str, series: pd.Series, etf_names: dict,
                    adv: pd.Series, raw_ret: pd.Series, bg: str) -> str:
    rows = ""
    for ticker, ratio in series.items():
        name     = etf_names.get(ticker, "—")[:55]
        arrow    = "&#9650;" if ratio > 0 else "&#9660;"
        adv_val  = adv.get(ticker, float("nan"))
        adv_disp = f"${adv_val/1e6:.0f}M" if not pd.isna(adv_val) else "—"
        r        = raw_ret.get(ticker, float("nan"))
        r_disp   = f"{r:+.1%}" if not pd.isna(r) else "—"
        rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='font-weight:bold'>{ticker}</td>"
            f"<td>{name}</td>"
            f"<td style='text-align:right'>{arrow}&nbsp;{ratio:+.2f}</td>"
            f"<td style='text-align:right'>{r_disp}</td>"
            f"<td style='text-align:right;color:#555'>{adv_disp}</td>"
            f"</tr>\n"
        )
    return f"""
<table cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;width:100%;margin-bottom:20px;">
<tr style="background:#333;color:white">
  <th colspan="5">{label}&nbsp;&nbsp;({len(series)} ETFs)</th>
</tr>
<tr style="background:#e8e8e8;font-size:12px">
  <th>Ticker</th><th>Name</th>
  <th style="text-align:right">252d Ret/Vol</th>
  <th style="text-align:right">252d Return</th>
  <th style="text-align:right">ADV (21d)</th>
</tr>
{rows}
</table>"""


def _emerging_table(label: str, series: pd.Series, etf_names: dict,
                    adv: pd.Series, raw_ret: pd.Series, bg: str) -> str:
    rows = ""
    for ticker, ratio in series.items():
        name     = etf_names.get(ticker, "—")[:55]
        arrow    = "&#9650;" if ratio > 0 else "&#9660;"
        adv_val  = adv.get(ticker, float("nan"))
        adv_disp = f"${adv_val/1e6:.0f}M" if not pd.isna(adv_val) else "—"
        r        = raw_ret.get(ticker, float("nan"))
        r_disp   = f"{r:+.1%}" if not pd.isna(r) else "—"
        rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='font-weight:bold'>{ticker}</td>"
            f"<td>{name}</td>"
            f"<td style='text-align:right'>{arrow}&nbsp;{ratio:+.2f}</td>"
            f"<td style='text-align:right'>{r_disp}</td>"
            f"<td style='text-align:right;color:#555'>{adv_disp}</td>"
            f"</tr>\n"
        )
    return f"""
<table cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;width:100%;margin-bottom:20px;">
<tr style="background:#555;color:white">
  <th colspan="5">{label}&nbsp;&nbsp;({len(series)} ETFs)</th>
</tr>
<tr style="background:#e8e8e8;font-size:12px">
  <th>Ticker</th><th>Name</th>
  <th style="text-align:right">100d Ret/Vol</th>
  <th style="text-align:right">100d Return</th>
  <th style="text-align:right">ADV (21d)</th>
</tr>
{rows}
</table>"""


def _reversal_table(label: str, series: pd.Series, etf_names: dict,
                    mom_long: set, mom_short: set, raw_5d: pd.Series, bg: str) -> str:
    rows = ""
    for ticker, ratio in series.items():
        name  = etf_names.get(ticker, "—")[:55]
        arrow = "&#9650;" if ratio > 0 else "&#9660;"
        if ticker in mom_long:
            bucket, bucket_color = "Mom LONG",  "#1E88E5"
        elif ticker in mom_short:
            bucket, bucket_color = "Mom SHORT", "#E53935"
        else:
            bucket, bucket_color = "Neutral",   "#9E9E9E"
        r     = raw_5d.get(ticker, float("nan"))
        r_disp = f"{r:+.2%}" if not pd.isna(r) else "—"
        rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='font-weight:bold'>{ticker}</td>"
            f"<td>{name}</td>"
            f"<td style='text-align:right'>{arrow}&nbsp;{ratio:+.2f}</td>"
            f"<td style='text-align:right'>{r_disp}</td>"
            f"<td style='text-align:center;color:{bucket_color};font-weight:bold'>{bucket}</td>"
            f"</tr>\n"
        )
    return f"""
<table cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;width:100%;margin-bottom:20px;">
<tr style="background:#333;color:white">
  <th colspan="5">{label}&nbsp;&nbsp;({len(series)} ETFs)</th>
</tr>
<tr style="background:#e8e8e8;font-size:12px">
  <th>Ticker</th><th>Name</th>
  <th style="text-align:right">5d / DailyStd</th>
  <th style="text-align:right">5d Return</th>
  <th style="text-align:center">Momentum Bucket</th>
</tr>
{rows}
</table>"""


def make_html(
    s_os: dict,
    s_is: dict,
    signal: pd.Series,
    signal_5d: pd.Series,
    signal_100d: pd.Series,
    etf_names: dict,
    tickers: list[str],
    adv: pd.Series,
    raw_252d: pd.Series,
    raw_100d: pd.Series,
    raw_5d: pd.Series,
    vix_prev: float,
    os_start: str,
) -> str:
    today_str = date.today().strftime("%B %d, %Y")

    # 252d momentum buckets
    q      = max(1, len(tickers) // 4)
    ranked = signal.reindex(tickers).dropna().rank(ascending=False)
    n_v    = len(ranked)
    longs  = signal.reindex(ranked[ranked <= q].index).sort_values(ascending=False)
    shorts = signal.reindex(ranked[ranked > n_v - q].index).sort_values()
    mom_long_set  = set(longs.index)
    mom_short_set = set(shorts.index)

    # 100d buckets — exclude names already in 252d top/bottom quartile
    sig100_clean = signal_100d.reindex(tickers).dropna()
    q100         = max(1, len(sig100_clean) // 4)
    ranked100    = sig100_clean.rank(ascending=False)
    n100         = len(ranked100)
    emerging_longs  = sig100_clean[
        (ranked100 <= q100) & (~sig100_clean.index.isin(mom_long_set))
    ].sort_values(ascending=False)
    emerging_shorts = sig100_clean[
        (ranked100 > n100 - q100) & (~sig100_clean.index.isin(mom_short_set))
    ].sort_values()

    # 5-day worst / best performers
    sig5_clean = signal_5d.reindex(tickers).dropna()
    n5         = max(1, len(sig5_clean) // 4)
    worst5     = sig5_clean.nsmallest(n5)
    best5      = sig5_clean.nlargest(n5)

    # VIX badge colour
    if pd.isna(vix_prev):
        vix_color, vix_label = "#888", "N/A"
    elif vix_prev > 20:
        vix_color, vix_label = "#E53935", f"{vix_prev:.1f} &#9650; (Reversal regime)"
    elif vix_prev < 15:
        vix_color, vix_label = "#1E88E5", f"{vix_prev:.1f} &#9660; (Momentum regime)"
    else:
        vix_color, vix_label = "#F9A825", f"{vix_prev:.1f} (Neutral / Cash zone)"

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;font-size:14px;color:#333;
             max-width:960px;margin:auto;padding:20px">

<h2 style="margin-bottom:2px">Momentum Strategy Monitor</h2>
<p style="margin-top:0;color:#666">
  {today_str}&nbsp;|&nbsp;{UNIVERSE}&nbsp;|&nbsp;
  252d lookback&nbsp;|&nbsp;weekly rebal&nbsp;|&nbsp;T+1 execution
</p>

<h3 style="border-bottom:2px solid #555;padding-bottom:4px">
  Market Regime
</h3>
<p style="font-size:16px">
  VIX (prev close):&nbsp;
  <strong style="color:{vix_color}">{vix_label}</strong>
</p>

<h3 style="border-bottom:2px solid #ff7f0e;padding-bottom:4px;color:#cc5500">
  Out-of-Sample Performance
  <span style="font-weight:normal;font-size:13px">(since {os_start})</span>
</h3>
{_stats_row(s_os, footnote=s_os["n_days"] < 20)}

<h3 style="border-bottom:2px solid #1f77b4;padding-bottom:4px;color:#1f5f99">
  In-Sample Performance
  <span style="font-weight:normal;font-size:13px">(backtest)</span>
</h3>
{_stats_row(s_is)}

<br>
<img src="cid:chart"
     style="width:100%;max-width:900px;border:1px solid #ddd"
     alt="IS/OS cumulative return chart">

<h3 style="margin-top:28px">
  252d Momentum — Current Book
  <span style="font-weight:normal;font-size:13px;color:#666">
    — based on latest available close
  </span>
</h3>
{_momentum_table("LONG — Top Quartile (252d)",     longs,  etf_names, adv, raw_252d, "#d4edda")}
{_momentum_table("SHORT — Bottom Quartile (252d)", shorts, etf_names, adv, raw_252d, "#f8d7da")}

<h3 style="margin-top:28px">
  100d Momentum — Emerging Signals
  <span style="font-weight:normal;font-size:13px;color:#666">
    — top/bottom quartile by 100d return, <em>excluding</em> names already in 252d book
  </span>
</h3>
{_emerging_table("Emerging LONG (100d top quartile, not yet in 252d longs)",   emerging_longs,  etf_names, adv, raw_100d, "#e8f5e9")}
{_emerging_table("Emerging SHORT (100d bottom quartile, not yet in 252d shorts)", emerging_shorts, etf_names, adv, raw_100d, "#fce4ec")}

<h3 style="margin-top:28px">
  5-Day Short-Term Performance
  <span style="font-weight:normal;font-size:13px;color:#666">
    — colour shows 252d momentum bucket
  </span>
</h3>
{_reversal_table("Worst 5-Day Performers (Bottom Quartile — by 5d Ret/Vol)",  worst5, etf_names, mom_long_set, mom_short_set, raw_5d, "#fff3cd")}
{_reversal_table("Best 5-Day Performers (Top Quartile — by 5d Ret/Vol)",      best5,  etf_names, mom_long_set, mom_short_set, raw_5d, "#d4edda")}

<p style="color:#bbb;font-size:11px;border-top:1px solid #eee;padding-top:8px">
  Prices via Yahoo Finance &nbsp;&bull;&nbsp;
  No look-ahead bias &nbsp;&bull;&nbsp;
  T+1 execution model
</p>
</body>
</html>"""


# ── Email ──────────────────────────────────────────────────────────────────────

def send_email(to: list[str], subject: str, html: str, chart_png: bytes) -> None:
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_user or not gmail_pass:
        raise RuntimeError(
            "Missing credentials.\n"
            "  Set GMAIL_USER and GMAIL_APP_PASSWORD environment variables.\n"
            "  App password: https://myaccount.google.com/apppasswords"
        )

    msg            = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = ", ".join(to)

    alt = MIMEMultipart("alternative")
    msg.attach(alt)
    alt.attach(MIMEText(html, "html", "utf-8"))

    img = MIMEImage(chart_png, name="chart.png")
    img.add_header("Content-ID", "<chart>")
    img.add_header("Content-Disposition", "inline", filename="chart.png")
    msg.attach(img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.sendmail(gmail_user, to, msg.as_string())
    print(f"  Sent -> {', '.join(to)}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily momentum monitor (GitHub Actions edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--to", nargs="+", default=RECIPIENTS)
    parser.add_argument("--os-start", default=OS_START)
    parser.add_argument("--preview",  action="store_true",
                        help="Save HTML + chart locally; do not send email")
    args = parser.parse_args()

    print("Loading universe ...")
    tickers = load_universe()
    print(f"  {len(tickers)} tickers")

    print("Loading prices (Yahoo Finance) ...")
    close, volume = load_prices_yf(tickers)
    available     = [t for t in tickers if t in close.columns]

    print("Loading VIX + SPY ...")
    vix     = load_vix_yf()
    spy_px  = load_spy_yf()
    spy_ret = spy_px.pct_change(fill_method=None).fillna(0)
    # previous trading day VIX (T-1 relative to latest price date)
    last_px_date = close.index[-1]
    vix_dates    = vix.index[vix.index <= last_px_date].tolist()
    vix_prev     = float(vix.loc[vix_dates[-2]]) if len(vix_dates) >= 2 else float("nan")
    print(f"  VIX prev close ({vix_dates[-2] if len(vix_dates)>=2 else 'N/A'}): {vix_prev:.1f}")

    print("Loading ETF names ...")
    etf_names = load_etf_names()

    print("Computing ADV ...")
    adv = compute_adv(close[available], volume[available])

    print("Running backtest ...")
    pnl, _ = run_backtest(close, available, LOOKBACK, REBAL_DAYS, SKIP)

    is_pnl = pnl[pnl.index < args.os_start]
    os_pnl = pnl[pnl.index >= args.os_start]
    s_is   = stats(is_pnl)
    s_os   = stats(os_pnl)

    print(f"  IS: {s_is['n_days']} days  Sharpe={s_is['sharpe']:+.2f}  "
          f"CumRet={s_is['cum_ret']:+.1%}")
    print(f"  OS: {s_os['n_days']} days  Sharpe={s_os['sharpe']:+.2f}  "
          f"CumRet={s_os['cum_ret']:+.1%}")

    print("Computing current signals ...")
    signal      = get_current_signal(close, available, LOOKBACK, SKIP)
    signal_5d   = get_5d_signal(close, available)
    signal_100d = get_100d_signal(close, available)
    raw_252d    = get_raw_return(close, available, LOOKBACK)
    raw_100d    = get_raw_return(close, available, 100)
    raw_5d      = get_raw_return(close, available, SIGNAL_DAYS)

    print("Generating chart ...")
    chart_png = make_chart(pnl, args.os_start)

    today_str = date.today().strftime("%Y-%m-%d")
    vix_tag   = f"VIX {vix_prev:.0f}" if not pd.isna(vix_prev) else "VIX N/A"
    subject   = (
        f"Momentum Monitor {today_str}"
        f" | {vix_tag}"
        f" | OS {s_os['cum_ret']:+.1%} ({s_os['n_days']}d)"
        f" | Sharpe {s_os['sharpe']:+.2f}"
    )
    html = make_html(s_os, s_is, signal, signal_5d, signal_100d, etf_names, available,
                     adv, raw_252d, raw_100d, raw_5d, vix_prev, args.os_start)

    out_dir = _HERE / "results_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        import base64
        chart_b64    = base64.b64encode(chart_png).decode("ascii")
        html_preview = html.replace(
            'src="cid:chart"',
            f'src="data:image/png;base64,{chart_b64}"',
        )
        html_path  = out_dir / f"monitor_preview_{today_str}.html"
        chart_path = out_dir / f"monitor_chart_{today_str}.png"
        html_path.write_text(html_preview, encoding="utf-8")
        chart_path.write_bytes(chart_png)
        print(f"  Preview -> {html_path}")
        print(f"  Chart   -> {chart_path}")
    else:
        print(f"Sending email to {', '.join(args.to)} ...")
        send_email(args.to, subject, html, chart_png)

    print("Done.")


if __name__ == "__main__":
    main()
