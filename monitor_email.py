#!/usr/bin/env python3
"""
Daily momentum strategy performance monitor.

Sends HTML email with:
  - OS stats since OS_START (default 2026-05-01)
  - IS stats for the full backtest period
  - IS/OS cumulative return chart (IS solid, OS dotted orange)
  - Current long and short ETF positions with names

Setup
-----
    Create a Gmail app password at https://myaccount.google.com/apppasswords
    Set environment variables:
        GMAIL_USER          your Gmail address
        GMAIL_APP_PASSWORD  the 16-char app password

Usage
-----
    python monitor_email.py                # send to rogerwugang@gmail.com
    python monitor_email.py --preview      # save HTML/chart locally, no send
    python monitor_email.py --to other@example.com
    python monitor_email.py --os-start 2026-04-01
"""

import argparse
import io
import os
import smtplib
import sys
from datetime import date
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent
DATA_DIR   = _HERE.parent.parent / "data"
DAILY_FILE = DATA_DIR / "daily/all.parquet"
CORR_DIR   = _HERE / "results_corr"
REF_FILE   = DATA_DIR / "reference/etf_tickers.parquet"

UNIVERSE   = "K60_hierarchical"
LOOKBACK   = 252
SKIP       = 0
REBAL_DAYS = 5
OS_START   = "2026-05-01"
RECIPIENT  = "rogerwugang@gmail.com"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    path = CORR_DIR / f"selected_{UNIVERSE}.csv"
    return pd.read_csv(path)["ticker"].tolist()


def load_prices(tickers: list[str]) -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_FILE, columns=["date", "ticker", "close"])
    daily = daily[daily["ticker"].isin(tickers) & daily["close"].notna() & daily["close"].gt(0)]
    close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    close.sort_index(inplace=True)
    return close.ffill()


def load_etf_names() -> dict[str, str]:
    ref = pd.read_parquet(REF_FILE, columns=["ticker", "name"])
    return ref.set_index("ticker")["name"].to_dict()


# ── Backtest + positions ───────────────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    tickers: list[str],
    lookback: int,
    rebal_days: int,
    skip: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (daily_pnl, current_positions).
    current_positions: the most-recently-set position vector (signed notional).
    """
    px = close[tickers].dropna(how="all")
    dates = px.index.tolist()
    if len(dates) < lookback + skip + rebal_days + 1:
        return pd.Series(dtype=float), pd.Series(0.0, index=tickers)

    daily_pnl = []
    positions = pd.Series(0.0, index=tickers)
    rebal_counter = 0

    for i in range(lookback + skip + 1, len(dates)):
        today    = dates[i]
        px_today = px.loc[today, tickers]
        prev_px  = px.iloc[i - 1][tickers]

        day_ret = (px_today - prev_px) / prev_px
        day_pnl = (positions * day_ret).sum()
        daily_pnl.append((today, day_pnl))

        if rebal_counter == 0:
            signal_px = px.iloc[i - 1 - skip][tickers]
            past_px   = px.iloc[i - 1 - skip - lookback][tickers]
            signal    = (signal_px - past_px) / past_px
            valid = signal.dropna()
            if len(valid) >= 4:
                ranked  = valid.rank(ascending=False)
                n_valid = len(valid)
                q       = max(1, n_valid // 4)
                longs   = valid[ranked <= q].index.tolist()
                shorts  = valid[ranked > n_valid - q].index.tolist()
                positions = pd.Series(0.0, index=tickers)
                if longs:  positions[longs]  =  1.0 / len(longs)
                if shorts: positions[shorts] = -1.0 / len(shorts)
            else:
                positions = pd.Series(0.0, index=tickers)

        rebal_counter = (rebal_counter + 1) % rebal_days

    pnl = pd.DataFrame(daily_pnl, columns=["date", "pnl"]).set_index("date")["pnl"]
    return pnl, positions


def get_current_signal(
    close: pd.DataFrame,
    tickers: list[str],
    lookback: int,
    skip: int,
) -> pd.Series:
    """252-day return ending `skip` days ago — used for position display."""
    px = close[tickers].dropna(how="all")
    dates = px.index.tolist()
    if len(dates) < lookback + skip + 1:
        return pd.Series(dtype=float)
    signal_px = px.iloc[-1 - skip] if skip > 0 else px.iloc[-1]
    past_px   = px.iloc[-1 - skip - lookback]
    return ((signal_px - past_px) / past_px).dropna()


# ── Statistics ─────────────────────────────────────────────────────────────────

def stats(pnl: pd.Series) -> dict:
    if pnl.empty:
        return {"ann_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0, "cum_ret": 0.0, "n_days": 0}
    cum  = (1 + pnl).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    ann  = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    sharpe = float(ann / vol) if vol > 0 else 0.0
    return {
        "ann_ret": float(ann),
        "sharpe":  sharpe,
        "max_dd":  float(dd.min()),
        "cum_ret": float(cum.iloc[-1] - 1),
        "n_days":  int(len(pnl)),
    }


# ── Chart ───────────────────────────────────────────────────────────────────────

def make_chart(pnl: pd.Series, os_start: str) -> bytes:
    """
    IS period: solid blue line.
    OS period: dotted orange line.
    Both plotted on a single continuous cumulative return axis.
    """
    cum = (1 + pnl).cumprod() - 1

    is_mask = pd.to_datetime(cum.index) < pd.to_datetime(os_start)
    os_mask = ~is_mask

    is_cum = cum[is_mask]
    os_cum = cum[os_mask]

    fig, ax = plt.subplots(figsize=(12, 5))

    if not is_cum.empty:
        ax.plot(
            pd.to_datetime(is_cum.index), is_cum.values * 100,
            color="#1f77b4", linewidth=1.5, label="IS (backtest)", zorder=3,
        )

    if not os_cum.empty:
        # Bridge: connect IS endpoint to OS start so the line is continuous
        bridge_dates = []
        bridge_vals  = []
        if not is_cum.empty:
            bridge_dates.append(pd.to_datetime(is_cum.index[-1]))
            bridge_vals.append(is_cum.iloc[-1] * 100)
        bridge_dates.append(pd.to_datetime(os_cum.index[0]))
        bridge_vals.append(os_cum.iloc[0] * 100)

        ax.plot(
            pd.to_datetime(os_cum.index), os_cum.values * 100,
            color="#ff7f0e", linewidth=2.0, linestyle="--",
            label=f"OS (live, from {os_start})", zorder=4,
        )

    ax.axvline(pd.to_datetime(os_start), color="gray", linewidth=0.9,
               linestyle=":", label="IS / OS split")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        f"{UNIVERSE}  |  252d momentum  |  weekly rebal\n"
        f"IS solid blue  /  OS dotted orange",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── HTML email body ────────────────────────────────────────────────────────────

def _stats_table(s: dict, note: str = "") -> str:
    sharpe_note = " *" if s["n_days"] < 20 else ""
    return f"""
<table cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%;margin-bottom:6px;">
<tr style="background:#f0f0f0;font-weight:bold">
  <th>Sharpe{sharpe_note}</th><th>Ann Return</th>
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
{"<p style='font-size:11px;color:#888'>* Sharpe not meaningful with fewer than 20 days</p>" if s['n_days'] < 20 else ""}
{note}
"""


def _position_table(label: str, series: pd.Series, etf_names: dict, bg: str) -> str:
    rows = ""
    for ticker, ret in series.items():
        name = etf_names.get(ticker, "—")[:55]
        arrow = "&#9650;" if ret > 0 else "&#9660;"
        rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='font-weight:bold'>{ticker}</td>"
            f"<td>{name}</td>"
            f"<td style='text-align:right'>{arrow} {ret:+.1%}</td>"
            f"</tr>\n"
        )
    return f"""
<table cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;width:100%;margin-bottom:20px;">
<tr style="background:#333;color:white">
  <th colspan="3">{label} &nbsp;({len(series)} ETFs)</th>
</tr>
<tr style="background:#e8e8e8;font-size:12px">
  <th>Ticker</th><th>Name</th><th style="text-align:right">252d Signal Return</th>
</tr>
{rows}
</table>
"""


def make_html(
    s_os: dict,
    s_is: dict,
    signal: pd.Series,
    etf_names: dict,
    tickers: list[str],
    os_start: str,
) -> str:
    today_str = date.today().strftime("%B %d, %Y")

    q      = max(1, len(tickers) // 4)
    ranked = signal.reindex(tickers).dropna().rank(ascending=False)
    n_v    = len(ranked)
    longs  = signal.reindex(ranked[ranked <= q].index).sort_values(ascending=False)
    shorts = signal.reindex(ranked[ranked > n_v - q].index).sort_values()

    long_tbl  = _position_table("LONG — Top Quartile",  longs,  etf_names, "#d4edda")
    short_tbl = _position_table("SHORT — Bottom Quartile", shorts, etf_names, "#f8d7da")

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;font-size:14px;color:#333;
             max-width:920px;margin:auto;padding:20px">

<h2 style="margin-bottom:2px">Momentum Strategy Monitor</h2>
<p style="margin-top:0;color:#666">{today_str} &nbsp;|&nbsp;
{UNIVERSE} &nbsp;|&nbsp; 252d lookback &nbsp;|&nbsp; weekly rebal &nbsp;|&nbsp; T+1 execution</p>

<h3 style="border-bottom:2px solid #ff7f0e;padding-bottom:4px;color:#cc5500">
  Out-of-Sample Performance &nbsp;<span style="font-weight:normal;font-size:13px">(since {os_start})</span>
</h3>
{_stats_table(s_os)}

<h3 style="border-bottom:2px solid #1f77b4;padding-bottom:4px;color:#1f5f99">
  In-Sample Performance &nbsp;<span style="font-weight:normal;font-size:13px">(backtest)</span>
</h3>
{_stats_table(s_is)}

<br>
<img src="cid:chart" style="width:100%;max-width:900px;border:1px solid #ddd"
     alt="IS/OS cumulative return chart">

<h3 style="margin-top:24px">Current Positions
  <span style="font-weight:normal;font-size:13px;color:#666">
    — based on latest available signal
  </span>
</h3>
{long_tbl}
{short_tbl}

<p style="color:#bbb;font-size:11px;border-top:1px solid #eee;padding-top:8px">
  Generated by monitor_email.py &nbsp;&bull;&nbsp; No look-ahead bias &nbsp;&bull;&nbsp; T+1 execution model
</p>
</body>
</html>"""


# ── Email sending ──────────────────────────────────────────────────────────────

def send_email(to: str, subject: str, html: str, chart_png: bytes) -> None:
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_user or not gmail_pass:
        raise RuntimeError(
            "Missing credentials.\n"
            "  export GMAIL_USER=you@gmail.com\n"
            "  export GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx\n"
            "Get an app password at: https://myaccount.google.com/apppasswords"
        )

    msg          = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = to

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
    print(f"  Sent -> {to}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily momentum strategy monitor email",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--to",       default=RECIPIENT, help="Recipient email address")
    parser.add_argument("--os-start", default=OS_START,  help="OS start date (YYYY-MM-DD)")
    parser.add_argument("--preview",  action="store_true",
                        help="Save HTML + chart locally; do not send email")
    args = parser.parse_args()

    print("Loading universe ...")
    tickers = load_universe()
    print(f"  {len(tickers)} tickers in {UNIVERSE}")

    print("Loading prices ...")
    close     = load_prices(tickers)
    available = [t for t in tickers if t in close.columns]
    print(f"  {len(close):,} days  {close.index[0]} -> {close.index[-1]}")
    print(f"  {len(available)} / {len(tickers)} tickers available")

    print("Loading ETF names ...")
    etf_names = load_etf_names()

    print("Running backtest ...")
    pnl, _ = run_backtest(close, available, LOOKBACK, REBAL_DAYS, SKIP)

    is_pnl = pnl[pnl.index < args.os_start]
    os_pnl = pnl[pnl.index >= args.os_start]
    s_is   = stats(is_pnl)
    s_os   = stats(os_pnl)

    print(f"  IS: {s_is['n_days']} days  Sharpe={s_is['sharpe']:+.2f}  CumRet={s_is['cum_ret']:+.1%}")
    print(f"  OS: {s_os['n_days']} days  Sharpe={s_os['sharpe']:+.2f}  CumRet={s_os['cum_ret']:+.1%}")

    print("Computing current signal ...")
    signal = get_current_signal(close, available, LOOKBACK, SKIP)

    print("Generating chart ...")
    chart_png = make_chart(pnl, args.os_start)

    today_str = date.today().strftime("%Y-%m-%d")
    subject   = (
        f"Momentum Monitor {today_str}"
        f" | OS {s_os['cum_ret']:+.1%} ({s_os['n_days']}d)"
        f" | Sharpe {s_os['sharpe']:+.2f}"
    )
    html = make_html(s_os, s_is, signal, etf_names, available, args.os_start)

    out_dir = _HERE / "results_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        html_path  = out_dir / f"monitor_preview_{today_str}.html"
        chart_path = out_dir / f"monitor_chart_{today_str}.png"
        html_path.write_text(html, encoding="utf-8")
        chart_path.write_bytes(chart_png)
        print(f"  Preview -> {html_path}")
        print(f"  Chart   -> {chart_path}")
    else:
        print(f"Sending email to {args.to} ...")
        send_email(args.to, subject, html, chart_png)

    print("Done.")


if __name__ == "__main__":
    main()
