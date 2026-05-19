#!/usr/bin/env python3
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE   = Path(__file__).parent
tickers = pd.read_csv(_HERE / "results_corr/selected_K100_hierarchical.csv")["ticker"].tolist()
out     = _HERE / "etf_names_K100_hierarchical.csv"

rows = []
for i, t in enumerate(tickers):
    try:
        info = yf.Ticker(t).info
        name = info.get("longName") or info.get("shortName") or "—"
    except Exception:
        name = "—"
    rows.append({"ticker": t, "name": name})
    print(f"  [{i+1:>3}/{len(tickers)}]  {t:<8}  {name[:60]}")

pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved -> {out}")
