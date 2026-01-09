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



def plot_multiple_roots_subplot(
    roots_dict: Dict[str, list],
    metric_col: Optional[str] = None,
    subfolder_col: str = "subfolder",
    param_keys: List[str] = DEFAULT_PARAM_KEYS,
    aggfunc: str = "mean",
    out_dir: str = "plots",
    cmap: str = "tab10",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> None:
    """
    For each prompt, creates a separate figure with subplots for each root group.
    Each figure is saved as a separate file per prompt.
    """
    import math
    import copy
    # Load all dataframes for each root group
    dfs_by_group = {}
    for group, roots in roots_dict.items():
        df = load_and_parse_from_folders(roots, subfolder_col=subfolder_col, param_keys=param_keys)
        if metric_col is None:
            metric = pick_last_cum_regret_col(df)
        else:
            metric = metric_col
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        # Normalize as in make_heatmaps_from_roots
        m = re.match(r"^cum_regret_(\d+)_mean$", metric)
        turns = int(m.group(1)) + 1 if m else 1
        if turns <= 0:
            turns = 1
        scale_max_regret = {}
        for scale_name, scale_values in SCALES.items():
            max_reward = max(scale_values)
            min_reward = min(scale_values)
            scale_max_regret[scale_name] = max_reward - min_reward
        def normalize_row(row):
            scale = row.get("scale")
            val = row.get(metric)
            if scale in scale_max_regret and pd.notnull(val):
                denom = scale_max_regret[scale]
                if denom != 0:
                    return val / (denom * turns)
            return val
        df[metric] = df.apply(normalize_row, axis=1)
        dfs_by_group[group] = (df, metric)

    # For each prompt, create a figure with subplots for each group
    for prompt in PROMPTS:
        # Check if any group has data for this prompt
        has_data = False
        for group, (df, metric) in dfs_by_group.items():
            if not df[df["prompt"].fillna("NA").astype(str) == prompt].empty:
                has_data = True
                break
        if not has_data:
            continue

        n = len(roots_dict)
        ncols = min(n, 2)
        nrows = math.ceil(n / ncols)
        width = 8 * ncols
        height = 4 * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        if n == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        legend_handles = None
        for idx, (group, (df, metric)) in enumerate(dfs_by_group.items()):
            # Only plot for this prompt
            df_prompt = df[df["prompt"].fillna("NA").astype(str) == prompt]
            if df_prompt.empty:
                axes[idx].set_visible(False)
                continue
            legend_handles = generate_grouped_bar_charts(
                df_prompt, metric, param_keys=param_keys, aggfunc=aggfunc, out_dir=out_dir, title=group, xlabel=xlabel, ylabel=ylabel, cmap=cmap, ax=axes[idx]
            )
            axes[idx].set_title(group)
        # Add legend to the right of the figure (not subplot)
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc='center left',
                bbox_to_anchor=(0.85, 0.5),
                borderaxespad=0.,
                frameon=True
            )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig_title = title + f" — {prompt}" if title else f"Prompt: {prompt}"
        fig.suptitle(fig_title, y=1.04)
        fname = os.path.join(out_dir, f"barchart_prompt_{prompt.replace(' ', '_')}.png")
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)

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


def generate_grouped_bar_charts(
    df: pd.DataFrame,
    metric_col: str,
    param_keys: List[str] = DEFAULT_PARAM_KEYS,
    aggfunc: str = "mean",
    out_dir: str = "plots",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "tab10",
    ax=None
):
    # Set up color and hatch patterns
    colors = sns.color_palette(cmap, len(ARM_NAMES))
    hatches = ["", "//", "////"][:len(SCALES)]
    scale_keys = list(SCALES.keys())

    # Build table: for each scale (column) get value per arm
    table = df.pivot_table(index="arm", columns="scale", values=metric_col, aggfunc=aggfunc)
    table = table.reindex(index=ARM_NAMES, columns=scale_keys)
    data = table.fillna(0).values
    x = np.arange(len(scale_keys))
    group_width = 0.8
    bar_width = group_width / len(ARM_NAMES)

    if ax is None:
        fig, ax_ = plt.subplots(figsize=(max(5, len(scale_keys) * 2.5), 2.5))
    else:
        ax_ = ax

    for arm_idx, arm in enumerate(ARM_NAMES):
        vals = [data[arm_idx, sidx] for sidx in range(len(scale_keys))]
        offsets = x - (group_width / 2) + arm_idx * bar_width + bar_width / 2
        bars = ax_.bar(offsets, vals, width=bar_width, color=colors[arm_idx], label=arm, edgecolor='black', linewidth=1.2)
        for sidx, b in enumerate(bars):
            if hatches[sidx]:
                b.set_hatch(hatches[sidx])
                b.set_edgecolor('black')

    max_line = ax_.axhline(1.0, color="red", linestyle=(0, (1, 5)), linewidth=2.5, label="Theoretical Maximum")
    ax_.set_xticks(x)
    ax_.set_xticklabels(scale_keys)
    ax_.set_title(title or (f"{metric_col} — prompt: {prompt}" if prompt else metric_col))
    ax_.set_xlabel(xlabel or "scale")
    if ylabel is not None:
        ax_.set_ylabel(ylabel)

    import matplotlib.patches as mpatches
    arm_patches = [mpatches.Patch(facecolor=colors[i], label=ARM_NAMES[i]) for i in range(len(ARM_NAMES))]
    hatch_patches = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatches[i] or '', label=scale_keys[i]) for i in range(len(scale_keys))]
    legend_handles = arm_patches + hatch_patches + [max_line]
    return legend_handles


def make_heatmaps_from_roots(roots: List[str],
                            metric_col: Optional[str] = None,
                            subfolder_col: str = "subfolder",
                            param_keys: List[str] = DEFAULT_PARAM_KEYS,
                            aggfunc: str = "mean",
                            out_dir: str = "plots",
                            cmap: str = "tab10",
                            title: Optional[str] = None,
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None) -> List[str]:
    df = load_and_parse_from_folders(roots, subfolder_col=subfolder_col, param_keys=param_keys)
    # pick last cum_regret if metric_col not provided
    if metric_col is None:
        metric_col = pick_last_cum_regret_col(df)
        if metric_col is None:
            raise KeyError("No 'cum_regret_{i}_mean' columns found to select as default metric. Provide --metric_col.")

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # Determine number of turns from the metric column name
    m = re.match(r"^cum_regret_(\d+)_mean$", metric_col)
    turns = int(m.group(1)) if m else 1
    if turns <= 0:
        turns = 1

    # Normalize regrets by max regret for each scale and divide by turns (per-turn average)
    scale_max_regret = {}
    for scale_name, scale_values in SCALES.items():
        max_reward = max(scale_values)
        min_reward = min(scale_values)
        scale_max_regret[scale_name] = max_reward - min_reward

    def normalize_row(row):
        scale = row.get("scale")
        val = row.get(metric_col)
        if scale in scale_max_regret and pd.notnull(val):
            denom = scale_max_regret[scale]
            if denom != 0:
                return val / (denom * turns)
        return val

    df[metric_col] = df.apply(normalize_row, axis=1)

    saved = generate_grouped_bar_charts(df, metric_col, param_keys=param_keys, aggfunc=aggfunc, out_dir=out_dir, title=title, xlabel=xlabel, ylabel=ylabel, cmap=cmap)
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse eval_results.csv files and generate arm x scale heatmaps.")
    p.add_argument("--roots", required=True, nargs='+', help="One or more root folders to search for eval_results.csv files")
    p.add_argument("--metric_col", default=None, help="Metric column to plot (default: last cum_regret_{i}_mean)")
    p.add_argument("--out_dir", default="heatmaps", help="Directory to save heatmaps")
    p.add_argument("--agg", default="mean", help="Aggregation function for duplicate cells (default: mean)")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap name for heatmaps")
    p.add_argument("--title", default=None, help="Custom title for plots")
    p.add_argument("--xlabel", default=None, help="Custom x-axis label")
    p.add_argument("--ylabel", default=None, help="Custom y-axis label")
    return p


if __name__ == "__main__":
    # Example usage for subplot dictionary:
    roots_dict = {
        "Farm": ["experiments/20251207_prompt_testing", "experiments/20251209_farm_sentiment"],
        "Abstract Bandit": ["experiments/20251209_abandit"]
    }
    plot_multiple_roots_subplot(
        roots_dict,
        metric_col=None,  # auto-pick last cum_regret
        out_dir="plots/barcharts",
        cmap="tab10",
        title="Effect of Scale and Nomenclature on Model Regret",
        xlabel="Reward Scale",
        ylabel="Percent of Possible Regret"
    )
