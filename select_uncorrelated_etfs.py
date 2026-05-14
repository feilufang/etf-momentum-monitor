#!/usr/bin/env python3
"""
Select the least-correlated ETF subsets from the top-200 liquid universe.

Two methods
-----------
hierarchical (default)
    Compute pairwise return correlations → angular distance matrix →
    Ward linkage → cut dendrogram to K clusters → pick most-liquid
    ETF from each cluster.  Guarantees one ETF per natural return-
    behaviour group, so the K names span the exposure space.

greedy
    Start with the ETF that has the lowest average |corr| to all others.
    Iteratively add the ETF that minimises the mean pairwise |corr|
    with the already-selected set.  Directly optimises the objective
    but is a local-greedy heuristic.

Inputs (same paths as simulate_loc_etf.py)
-------------------------------------------
    data/daily/all.parquet       OHLCV for all tickers
    data/etf_universe.parquet    rolling top-200 per day

Outputs (in <output_dir>/)
--------------------------
    selected_K<N>_<method>.csv   ticker list with cluster / rank info
    corr_heatmap_K<N>.png        (if --plot flag)
    summary.csv                  avg pairwise |corr| for every run

Usage
-----
    python select_uncorrelated_etfs.py
    python select_uncorrelated_etfs.py --k 20 40 60 --method hierarchical
    python select_uncorrelated_etfs.py --lookback 126 --min-coverage 0.8
    python select_uncorrelated_etfs.py --method greedy --plot
    python select_uncorrelated_etfs.py --method both --min-vol 0.10 --plot
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE      = Path(__file__).parent
DATA_DIR   = _HERE.parent.parent / "data"
DAILY_FILE = DATA_DIR / "daily/all.parquet"
UNIV_FILE  = DATA_DIR / "etf_universe.parquet"
REF_FILE   = DATA_DIR / "reference/etf_tickers.parquet"

# Matches leveraged and inverse ETFs by name
_LEV_RE = re.compile(
    r"\b[23][xX]\b|UltraPro|ProShares Ultra|Direxion Dail|Daily Target",
    re.IGNORECASE,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_returns(
    lookback: int, min_coverage: float, min_vol: float, min_adv: float,
    exclude_leveraged: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns
        ret_wide   : (date × ticker) daily log-return matrix, NaN where missing
        avg_dv     : ticker → mean dollar volume over the lookback window
                     (used to break ties when selecting cluster reps)
    """
    print("Loading ETF universe ...")
    univ = pd.read_parquet(UNIV_FILE)
    etf_tickers = set(univ["ticker"].unique())
    print(f"  {len(etf_tickers):,} unique ETF tickers in universe")

    if exclude_leveraged:
        ref = pd.read_parquet(REF_FILE, columns=["ticker", "name"])
        lev = set(ref.loc[ref["name"].str.contains(_LEV_RE, na=False), "ticker"])
        before = len(etf_tickers)
        etf_tickers -= lev
        print(f"  {len(lev):,} leveraged/inverse ETFs identified, "
              f"{before - len(etf_tickers):,} removed from universe")

    print("Loading daily bars ...")
    daily = pd.read_parquet(
        DAILY_FILE, columns=["date", "ticker", "close", "vwap", "volume"]
    )
    daily = daily[daily["ticker"].isin(etf_tickers)]
    daily = daily[daily["close"].notna() & daily["close"].gt(0)]
    daily["dv"] = daily["vwap"] * daily["volume"]
    daily = daily.sort_values("date")

    # Keep only the most recent `lookback` trading dates
    all_dates = sorted(daily["date"].unique())
    if len(all_dates) > lookback:
        cutoff = all_dates[-lookback]
        daily = daily[daily["date"] >= cutoff]
        all_dates = all_dates[-lookback:]

    print(f"  Using {len(all_dates):,} trading days ending {all_dates[-1]}")

    close_wide = daily.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    )
    dv_wide = daily.pivot_table(
        index="date", columns="ticker", values="dv", aggfunc="sum"
    )

    # Drop tickers with insufficient coverage
    min_days = int(min_coverage * len(all_dates))
    coverage = close_wide.notna().sum()
    keep = coverage[coverage >= min_days].index
    close_wide = close_wide[keep]
    print(f"  {len(keep):,} ETFs pass ≥{min_coverage:.0%} coverage filter")

    ret_wide = np.log(close_wide).diff()
    avg_dv   = dv_wide[keep].mean()

    # Drop cash-like ETFs whose annualised vol is below the threshold
    ann_vol = ret_wide.std() * np.sqrt(252)
    after_vol = ann_vol[ann_vol >= min_vol].index
    dropped_vol = len(keep) - len(after_vol)
    print(f"  {len(after_vol):,} ETFs pass ≥{min_vol:.0%} annualised-vol filter "
          f"({dropped_vol:,} cash-like dropped)")

    # Drop ETFs below the minimum average daily dollar volume
    after_adv = avg_dv[after_vol][avg_dv[after_vol] >= min_adv].index
    dropped_adv = len(after_vol) - len(after_adv)
    print(f"  {len(after_adv):,} ETFs pass ≥${min_adv/1e6:.0f}M ADV filter "
          f"({dropped_adv:,} illiquid dropped)")

    return ret_wide[after_adv], avg_dv[after_adv]


# ── Correlation / distance ────────────────────────────────────────────────────

def build_corr_and_dist(ret_wide: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (corr_df, condensed_distance_array)."""
    corr = ret_wide.corr(method="pearson").clip(-1, 1)

    # Angular distance: d = sqrt(0.5*(1-r)) — a proper metric on [-1,1]
    dist_sq = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist_sq, 0.0)
    dist_condensed = squareform(dist_sq, checks=False)
    return corr, dist_condensed


# ── Method 1: hierarchical clustering ────────────────────────────────────────

def select_hierarchical(
    corr: pd.DataFrame,
    dist_condensed: np.ndarray,
    avg_dv: pd.Series,
    k: int,
) -> pd.DataFrame:
    tickers = corr.columns.tolist()

    Z = linkage(dist_condensed, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")

    rows = []
    for cluster_id in range(1, k + 1):
        members = [t for t, l in zip(tickers, labels) if l == cluster_id]
        if not members:
            continue
        # Pick the most liquid ETF in this cluster
        rep = avg_dv.reindex(members).idxmax()
        rows.append({
            "ticker":     rep,
            "cluster":    cluster_id,
            "cluster_size": len(members),
            "cluster_members": ",".join(sorted(members)),
            "avg_dv_rank": avg_dv.rank(ascending=False)[rep],
        })

    result = pd.DataFrame(rows).sort_values("avg_dv_rank").reset_index(drop=True)
    result.index.name = "selection_rank"
    return result


# ── Method 2: greedy min-correlation ─────────────────────────────────────────

def select_greedy(
    corr: pd.DataFrame,
    avg_dv: pd.Series,
    k: int,
) -> pd.DataFrame:
    tickers = corr.columns.tolist()
    corr_arr = corr.values.copy()
    abs_corr = np.abs(corr_arr)
    n = len(tickers)

    # Seed: ETF with the lowest mean |corr| to all others (excluding self)
    np.fill_diagonal(abs_corr, np.nan)
    mean_abs = np.nanmean(abs_corr, axis=1)
    np.fill_diagonal(abs_corr, 0.0)

    selected_idx = [int(np.nanargmin(mean_abs))]

    while len(selected_idx) < k:
        remaining = [i for i in range(n) if i not in selected_idx]
        # For each candidate, compute mean |corr| to already-selected set
        sel_arr = abs_corr[np.ix_(remaining, selected_idx)]
        mean_to_sel = sel_arr.mean(axis=1)
        best_pos = int(np.argmin(mean_to_sel))
        selected_idx.append(remaining[best_pos])

    rows = []
    for rank, idx in enumerate(selected_idx):
        rows.append({
            "ticker":       tickers[idx],
            "selection_rank": rank,
            "mean_abs_corr_to_set": (
                np.abs(corr_arr[idx, selected_idx]).mean()
                if rank > 0 else 0.0
            ),
            "avg_dv_rank": avg_dv.rank(ascending=False)[tickers[idx]],
        })

    return pd.DataFrame(rows)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def avg_pairwise_abs_corr(corr: pd.DataFrame, tickers: list[str]) -> float:
    sub = corr.loc[tickers, tickers].values
    n = len(tickers)
    upper = sub[np.triu_indices(n, k=1)]
    return float(np.mean(np.abs(upper)))


def plot_heatmap(corr: pd.DataFrame, tickers: list[str], title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  (skipping heatmap — install matplotlib + seaborn)")
        return

    sub = corr.loc[tickers, tickers]
    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 0.4),
                                    max(6, len(tickers) * 0.4)))
    sns.heatmap(
        sub, annot=len(tickers) <= 30, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.3,
        ax=ax, cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved heatmap -> {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select least-correlated ETF subsets from the top-200 universe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k", type=int, nargs="+", default=[20, 40, 60],
                        metavar="K", help="Subset sizes to compute")
    parser.add_argument("--method", choices=["hierarchical", "greedy", "both"],
                        default="hierarchical")
    parser.add_argument("--lookback", type=int, default=252,
                        help="Trading days of history to use for correlations")
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="Fraction of lookback days a ticker must have data for")
    parser.add_argument("--min-vol", type=float, default=0.10,
                        help="Minimum annualised daily return volatility (drop cash-like ETFs)")
    parser.add_argument("--min-adv", type=float, default=10e6,
                        help="Minimum average daily dollar volume in USD (drop illiquid ETFs)")
    parser.add_argument("--no-leveraged", action="store_true", default=True,
                        help="Exclude leveraged and inverse ETFs (default: on)")
    parser.add_argument("--include-leveraged", dest="no_leveraged",
                        action="store_false",
                        help="Include leveraged and inverse ETFs")
    parser.add_argument("--plot", action="store_true",
                        help="Save correlation heatmaps for each selected subset")
    parser.add_argument("--output", default=str(_HERE / "results_corr"))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = (["hierarchical", "greedy"] if args.method == "both"
               else [args.method])

    # ── Load data ──────────────────────────────────────────────────────────────
    ret_wide, avg_dv = load_returns(
        args.lookback, args.min_coverage, args.min_vol, args.min_adv, args.no_leveraged
    )
    n_etfs = len(ret_wide.columns)
    print(f"\n{n_etfs:,} ETFs in correlation matrix")

    print("Computing correlation matrix ...")
    corr, dist_condensed = build_corr_and_dist(ret_wide)

    # Baseline: avg pairwise |corr| across all ETFs
    baseline_corr = avg_pairwise_abs_corr(corr, corr.columns.tolist())
    print(f"  Baseline avg |corr| (all {n_etfs} ETFs): {baseline_corr:.4f}\n")

    summary_rows = [{"k": "all", "method": "—",
                     "avg_pairwise_abs_corr": baseline_corr,
                     "n_etfs": n_etfs}]

    # ── Run selections ─────────────────────────────────────────────────────────
    for method in methods:
        for k in sorted(args.k):
            if k > n_etfs:
                print(f"  Skipping K={k}: only {n_etfs} ETFs available")
                continue

            print(f"[{method}] K={k} ...")
            if method == "hierarchical":
                result = select_hierarchical(corr, dist_condensed, avg_dv, k)
            else:
                result = select_greedy(corr, avg_dv, k)

            selected = result["ticker"].tolist()
            mean_corr = avg_pairwise_abs_corr(corr, selected)

            print(f"  avg pairwise |corr|: {mean_corr:.4f}  "
                  f"(vs {baseline_corr:.4f} baseline, "
                  f"{(mean_corr / baseline_corr - 1) * 100:+.1f}%)")
            print(f"  Selected: {', '.join(selected)}\n")

            fname = out_dir / f"selected_K{k:02d}_{method}.csv"
            result.to_csv(fname, index=False)
            print(f"  Saved -> {fname}")

            if args.plot:
                title = f"Correlation heatmap  K={k}  [{method}]"
                plot_heatmap(corr, selected, title,
                             out_dir / f"corr_heatmap_K{k:02d}_{method}.png")

            summary_rows.append({
                "k":                   k,
                "method":              method,
                "avg_pairwise_abs_corr": mean_corr,
                "n_etfs":              len(selected),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSummary -> {out_dir / 'summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
