"""Create grouped bar charts from combined_regrets.csv.

Produces one figure per `prompt`. Each figure contains one subplot per `dataset`.
Bars are grouped by `scale` along the x-axis and colored by `arm`.
The plotted metric is the last `cum_regret_{i}_mean` column, normalized by the
maximum possible regret for the corresponding `scale` (see `SCALES` below)
and divided by the number of turns (i+1) to produce a per-turn fraction.

Saves PNG files to `plots/barcharts` by default.
"""
from __future__ import annotations

import os
import re
import argparse
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Known arms and scales used in the experiment datasets
ARM_NAMES = [
    "alphanumeric",
    "sem_rel_helpful",
    "sem_rel_mislead",
    "sent_helpful",
    "sent_mislead",
]

SCALES: Dict[str, list] = {
    "high_scale": [200, 100, 50],
    "low_scale": [2, 1, 0.5],
    "high_neg_scale": [-200, -100, -50],
}


def pick_last_cum_regret_col(df: pd.DataFrame) -> Optional[str]:
    pattern = re.compile(r"^cum_regret@?(\d+)_mean$")
    max_i = -1
    best = None
    for c in df.columns:
        m = pattern.match(c)
        if m:
            i = int(m.group(1))
            if i > max_i:
                max_i = i
                best = c
    return best


def normalize_final_turn(df: pd.DataFrame, metric_col: str) -> pd.Series:
    # Determine number of turns from the metric column name
    m = re.match(r"^cum_regret@?(\d+)_mean$", metric_col)
    turns = int(m.group(1)) + 1 if m else 1
    # Compute per-scale max regret (max - min) as in other scripts
    scale_max_regret = {}
    for scale_name, vals in SCALES.items():
        scale_max_regret[scale_name] = max(vals) - min(vals)

    def _norm(row):
        scale = row.get("scale")
        val = row.get(metric_col)
        if pd.isnull(val):
            return np.nan
        if scale in scale_max_regret and scale_max_regret[scale] != 0:
            return val / (scale_max_regret[scale] * turns)
        return val
    
    # import pdb; pdb.set_trace()

    return df.apply(_norm, axis=1)


def make_grouped_barchart(df: pd.DataFrame, metric_col: str, out_path: str, title: Optional[str] = None):
    # Ensure metric numeric
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # For each prompt, create a figure
    prompts = sorted(df["prompt"].dropna().unique())
    datasets = sorted(df["dataset"].dropna().unique())

    os.makedirs(out_path, exist_ok=True)

    colors = sns.color_palette("tab10", len(ARM_NAMES))
    hatches = ["", "//", "\\\\"][: len(SCALES)]
    scale_keys = list(SCALES.keys())

    for prompt in prompts:
        df_prompt = df[df["prompt"] == prompt]
        if df_prompt.empty:
            continue

        ncols = max(1, len(datasets))
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.5), squeeze=False)
        axes = axes.flatten()

        for ax_idx, dataset in enumerate(datasets):
            ax = axes[ax_idx]
            df_ds = df_prompt[df_prompt["dataset"] == dataset]
            if df_ds.empty:
                ax.set_visible(False)
                continue

            # Aggregate by arm x scale (mean across other dims if needed)
            table = (
                df_ds
                .groupby(["arm", "scale"])[metric_col]
                .mean()
                .unstack(fill_value=0)
                .reindex(index=ARM_NAMES, columns=scale_keys)
                .fillna(0)
            )

            data = table.values
            x = np.arange(len(scale_keys))
            group_width = 0.8
            bar_width = group_width / max(1, len(ARM_NAMES))

            for arm_idx, arm in enumerate(ARM_NAMES):
                offsets = x - (group_width / 2) + arm_idx * bar_width + bar_width / 2
                vals = data[arm_idx, :]
                bars = ax.bar(offsets, vals, width=bar_width, color=colors[arm_idx], label=arm, edgecolor="black", linewidth=1.0)
                # apply hatching by scale column
                for sidx, b in enumerate(bars):
                    hatch = hatches[sidx] if sidx < len(hatches) else ""
                    if hatch:
                        b.set_hatch(hatch)

            ax.axhline(1.0, color="red", linestyle=(0, (4, 4)), linewidth=1.0, label="Theoretical Maximum")
            # Add dashed black line at 0.208 for Structure Oracle
            ax.axhline(0.166, color="firebrick", linestyle=(0, (4, 4)), linewidth=1.0, label="Structure Oracle")
            ax.set_xticks(x)
            ax.set_xticklabels(scale_keys)
            ax.set_title(dataset.title())
            ax.set_xlabel("Reward Scale")
            ax.set_ylabel("Percent of Possible Regret")

        # Add a single legend on the right
        handles, labels = axes[0].get_legend_handles_labels()
        # Remove duplicate labels (e.g., if multiple axhline calls)
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                unique_handles.append(h)
                unique_labels.append(l)
                seen.add(l)
        fig.legend(unique_handles, unique_labels, loc="center right", bbox_to_anchor=(1.1, 0.5))
        fig.suptitle(((title or "") + (f" — {prompt}" if prompt else "")).title(), y=1.02)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        fname = os.path.join(out_path, f"barchart_prompt_{prompt}.png")
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make grouped barcharts from combined_regrets.csv")
    p.add_argument("--csv", default="combined_regrets.csv", help="Path to combined_regrets.csv")
    p.add_argument("--out_dir", default="plots/barcharts", help="Output directory for barcharts")
    p.add_argument("--metric_col", default=None, help="Metric column to plot (default: last cum_regret_*_mean)")
    p.add_argument("--model_scale", default="14B", help="Filter rows by model_scale (set empty string to skip filtering)")
    p.add_argument("--title", default="Effect of Scale and Nomenclature on Model Regret", help="Figure title prefix")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)

    # Optionally filter by model_scale
    if getattr(args, "model_scale", None) not in (None, ""):
        if "model_scale" in df.columns:
            df = df[df["model_scale"] == args.model_scale]
        else:
            raise KeyError("Column 'model_scale' not found in CSV to filter by model_scale")

    metric = args.metric_col or pick_last_cum_regret_col(df)
    if metric is None:
        raise KeyError("No cum_regret_*_mean column found in CSV. Provide --metric_col.")

    # Normalize final turn per scale
    df[metric] = normalize_final_turn(df, metric)

    make_grouped_barchart(df, metric, args.out_dir, title=args.title)


if __name__ == "__main__":
    main()
# Fill in code here