"""
Utility to load `eval_results.csv` files from a folder tree, parse the
`subfolder` strings into parameter columns (including `arm` and `scale`),
and generate arm x scale heatmaps for the last `cum_regret_{step}_mean` metric.

Usage (example):
  python scripts/parse_and_plot_heatmaps.py --root experiments/20251207_prompt_testing --out_dir plots/heatmaps

The script will search recursively for files named `eval_results.csv` under
the provided root, concatenate them, parse `subfolder`, automatically pick
the highest-index `cum_regret_{i}_mean` column as the metric (unless overridden),
and save heatmaps grouped by the other parsed parameters (e.g., `prompt`).
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

DEFAULT_PARAM_KEYS = ["model_name", "time_horizon", "think_budget", "prompt", "agent"]

# Hardcoded known values (user-provided)
ARM_NAMES = [
    "alphanumeric",
    "sem_rel_helpful",
    "sem_rel_mislead",
    "sent_helpful",
    "sent_mislead",
]

SCALES = {
    "high_scale": [200,100,50],
    "low_scale": [2,1,0.5],
    "high_neg_scale": [-200,-100,-50],
}

PROMPTS = ["fullhist", "actionhist", "summhist", "pre-budget"]


def find_eval_csvs(roots: List[str]) -> List[str]:
    """Recursively find files named `eval_results.csv` under each path in `roots`.

    `roots` may be a list of folder paths. Returns a list of matching file paths.
    """
    matches: List[str] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if "eval_results.csv" in filenames:
                matches.append(os.path.join(dirpath, "eval_results.csv"))
    # deduplicate while preserving order
    seen = set()
    unique = []
    for p in matches:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def parse_subfolder(subfolder: str, param_keys: List[str] = DEFAULT_PARAM_KEYS) -> Dict[str, Optional[str]]:
    """Parse `subfolder` only by looking for predefined ARM_NAMES, SCALES, and PROMPTS.

    This version intentionally does not attempt to parse other parameters. Any
    params not found remain `None`.
    """
    res: Dict[str, Optional[str]] = {k: None for k in param_keys}
    # include explicit prompt key as it's used for grouping
    res.update({"arm": None, "scale": None, "prompt": None, "raw": subfolder})

    if not isinstance(subfolder, str) or subfolder == "":
        return res

    s = subfolder

    # Find scale and arm by simple substring membership (no broad regex)
    for sc in SCALES:
        if sc in s:
            res["scale"] = sc
            break

    for a in ARM_NAMES:
        if a in s:
            res["arm"] = a
            break

    for pval in PROMPTS:
        if pval in s:
            res["prompt"] = "pre-budget" if pval in ("pre_budget", "pre-budget") else pval
            break

    return res

    return res


def load_and_parse_from_folders(roots: List[str], subfolder_col: str = "subfolder", param_keys: List[str] = DEFAULT_PARAM_KEYS) -> pd.DataFrame:
    """Load all `eval_results.csv` found under the provided `roots`, parse their `subfolder` strings,
    and return a concatenated DataFrame with parsed columns added.
    """
    files = find_eval_csvs(roots)
    if not files:
        raise FileNotFoundError(f"No eval_results.csv files found under: {roots}")

    dfs = []
    for p in files:
        df = pd.read_csv(p)
        if subfolder_col not in df.columns:
            raise KeyError(f"Column '{subfolder_col}' not found in {p}. Available columns: {list(df.columns)}")
        df = df.copy()
        parsed = df[subfolder_col].fillna("").astype(str).apply(lambda x: parse_subfolder(x, param_keys))
        parsed_df = pd.json_normalize(parsed)
        df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
        df["_source_csv"] = os.path.basename(p)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True, sort=False)
    return full


def pick_last_cum_regret_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the `cum_regret_{i}_mean` column with the largest i present in `df`.
    Returns None if no matching columns found.
    """
    pattern = re.compile(r"^cum_regret_(\d+)_mean$")
    max_i = -1
    best_col = None
    for c in df.columns:
        m = pattern.match(c)
        if m:
            i = int(m.group(1))
            if i > max_i:
                max_i = i
                best_col = c
    return best_col


def generate_arm_scale_heatmaps(df: pd.DataFrame,
                                metric_col: str,
                                param_keys: List[str] = DEFAULT_PARAM_KEYS,
                                aggfunc: str = "mean",
                                out_dir: str = "heatmaps",
                                cmap: str = "viridis") -> List[str]:
    """Generate and save heatmaps (arm x scale) for each unique combination of other params.

    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    if metric_col not in df.columns:
        raise KeyError(f"Metric column '{metric_col}' not found in DataFrame. Columns: {list(df.columns)}")
    if "arm" not in df.columns or "scale" not in df.columns:
        raise KeyError("Parsed columns 'arm' and/or 'scale' not found in DataFrame. Run parser first.")

    # Determine other keys to group by (present and not all-null)
    other_keys = [k for k in param_keys if k in df.columns and not df[k].isnull().all()]

    # If none, create a single group
    if other_keys:
        group_values = df[other_keys].fillna("NA").astype(str)
        unique_groups = group_values.drop_duplicates()
    else:
        unique_groups = pd.DataFrame([[]])

    # Prepare subplots
    if other_keys:
        iter_rows = list(unique_groups.itertuples(index=False, name=None))
    else:
        iter_rows = [tuple()]

    n_groups = len(iter_rows)
    ncols = min(3, n_groups)
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4), squeeze=False)

    # Compute global min/max for consistent color scale
    global_min = df[metric_col].min()
    global_max = df[metric_col].max()

    plotted = 0
    group_labels = []
    for idx, vals in enumerate(iter_rows):
        if other_keys:
            mask = np.ones(len(df), dtype=bool)
            for k, v in zip(other_keys, vals):
                mask &= df[k].fillna("NA").astype(str) == v
            subset = df[mask]
            group_label = "_".join([f"{k}-{v}" for k, v in zip(other_keys, vals)])
        else:
            subset = df
            group_label = "all"

        if subset.empty:
            continue

        # pivot table (aggregate duplicates)
        pivot = subset.pivot_table(index="arm", columns="scale", values=metric_col, aggfunc=aggfunc)
        pivot = pivot.reindex(index=ARM_NAMES, columns=list(SCALES.keys()))

        if pivot.isnull().all(axis=None):
            continue

        annot_df = pivot.copy()
        annot_vals = annot_df.round(3).astype(object)
        annot_vals = annot_vals.where(~annot_vals.isna(), '')

        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        sns.heatmap(
            pivot,
            annot=annot_vals,
            fmt='',
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            vmin=global_min,
            vmax=global_max,
            ax=ax
        )
        ax.set_title(group_label)
        ax.set_xlabel("scale")
        # Only show y-label for first subplot, drop for others
        if idx == 0:
            ax.set_ylabel("arm")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        group_labels.append(group_label)
        plotted += 1

    # Remove unused axes
    for idx in range(plotted, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row][col])

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    # Use the last plotted heatmap for colorbar
    norm = plt.Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=metric_col)

    plt.suptitle(f"{metric_col} — arm x scale heatmaps", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    fname = os.path.join(out_dir, f"heatmap_{metric_col}_combined.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    return [fname]


def make_heatmaps_from_roots(roots: List[str],
                            metric_col: Optional[str] = None,
                            subfolder_col: str = "subfolder",
                            param_keys: List[str] = DEFAULT_PARAM_KEYS,
                            aggfunc: str = "mean",
                            out_dir: str = "heatmaps",
                            cmap: str = "viridis") -> List[str]:
    df = load_and_parse_from_folders(roots, subfolder_col=subfolder_col, param_keys=param_keys)
    # pick last cum_regret if metric_col not provided
    if metric_col is None:
        metric_col = pick_last_cum_regret_col(df)
        if metric_col is None:
            raise KeyError("No 'cum_regret_{i}_mean' columns found to select as default metric. Provide --metric_col.")

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # Normalize regrets by max regret for each scale
    # Build a mapping from scale name to max regret
    scale_max_regret = {}
    for scale_name, scale_values in SCALES.items():
        max_reward = max(scale_values)
        min_reward = min(scale_values)
        scale_max_regret[scale_name] = max_reward - min_reward

    # Apply normalization to the metric column
    def normalize_row(row):
        scale = row.get("scale")
        val = row.get(metric_col)
        if scale in scale_max_regret and pd.notnull(val):
            denom = scale_max_regret[scale]
            if denom != 0:
                return val / denom
        return val

    df[metric_col] = df.apply(normalize_row, axis=1)

    saved = generate_arm_scale_heatmaps(df, metric_col, param_keys=param_keys, aggfunc=aggfunc, out_dir=out_dir, cmap=cmap)
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse eval_results.csv files and generate arm x scale heatmaps.")
    p.add_argument("--roots", required=True, nargs='+', help="One or more root folders to search for eval_results.csv files")
    p.add_argument("--metric_col", default=None, help="Metric column to plot (default: last cum_regret_{i}_mean)")
    p.add_argument("--out_dir", default="heatmaps", help="Directory to save heatmaps")
    p.add_argument("--agg", default="mean", help="Aggregation function for duplicate cells (default: mean)")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap name for heatmaps")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    saved = make_heatmaps_from_roots(args.roots, metric_col=args.metric_col, aggfunc=args.agg, out_dir=args.out_dir, cmap=args.cmap)
    if saved:
        print("Saved heatmaps:")
        for s in saved:
            print(" -", s)
    else:
        print("No heatmaps were created (no data or pivot empty).")
