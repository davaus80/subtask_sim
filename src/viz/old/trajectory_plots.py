"""Visualization helpers for exploration trajectories."""
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_exploration_trajectories(
    replicate_df: pd.DataFrame,
    variable: str,
    colour_map: Dict[Any, str],
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.05,
    linewidth: float = 0.5,
    figsize: tuple = (7, 4),
    legend: bool = True,
):
    """Plot exploration_count trajectories across turns for all rows in `replicate_df`.

    Parameters
    - replicate_df: DataFrame containing rows to plot. Must have columns named
      `exploration_count_0`, `exploration_count_1`, ...
    - variable: column name in `replicate_df` whose values determine colours.
    - colour_map: mapping from variable values to colour strings (e.g. '#rrggbb' or named colours).
    - ax: optional Matplotlib Axes to plot onto. If None, a new figure+axes are created.
    - alpha: line opacity for each trajectory (use low values for overlapping density).
    - linewidth: width of individual lines.
    - figsize: figure size used when creating a new figure.
    - legend: whether to draw a legend mapping variable values to colours.

    Returns the `ax` used for plotting.
    """
    if variable not in replicate_df.columns:
        raise KeyError(f"variable '{variable}' not found in replicate_df columns")

    # find exploration_count columns and sort by turn index
    expl_cols = [c for c in replicate_df.columns if c.startswith("exploration_count_")]
    if not expl_cols:
        raise ValueError("no columns starting with 'exploration_count_' found in replicate_df")

    def _turn_index(col: str):
        try:
            return int(col.rsplit("_", 1)[1])
        except Exception:
            return col

    expl_cols = sorted(expl_cols, key=_turn_index)
    turns: Iterable[int] = [int(c.rsplit("_", 1)[1]) for c in expl_cols]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # group rows by variable value and plot each trajectory with the mapped colour
    # use numpy arrays for slightly faster iteration
    grouped = replicate_df.groupby(variable)

    for val, group in grouped:
        color = colour_map.get(val, None)
        if color is None:
            color = "#888888"

        # convert to float to avoid plotting object-dtype issues
        try:
            data = group[expl_cols].to_numpy(dtype=float)
        except Exception:
            data = group[expl_cols].astype(float).to_numpy()

        # plot each row; set low alpha so overlapping density shows as darkness
        for row in data:
            ax.plot(turns, row, color=color, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel("turn")
    ax.set_ylabel("exploration_count")
    ax.set_title("Exploration trajectories")

    if legend:
        # create legend proxies in the same order as colour_map keys
        from matplotlib.lines import Line2D

        handles = []
        labels = []
        for k, col in colour_map.items():
            handles.append(Line2D([0], [0], color=col, lw=2))
            labels.append(str(k))

        if handles:
            ax.legend(handles, labels, title=variable, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.autoscale(enable=True, axis='y', tight=False)
    return ax


def compute_time_between_increases(
    replicate_df: pd.DataFrame,
    expl_prefix: str = "exploration_count_",
    from_val: float = 1.0,
    mid_val: float = 2.0,
    to_val: float = 3.0,
    inplace: bool = False,
):
    """Compute durations between transitions `from_val->mid_val` and the following `mid_val->to_val`.

    Adds/returns columns per-row:
    - `never_reaches_2`: True if the trajectory never reaches `mid_val` via a direct transition.
    - `never_reaches_3`: True if the trajectory never reaches `to_val` after first reaching `mid_val`.
    - `avg_turns_between_12_23`: average number of turns between each observed `from->mid` and the subsequent `mid->to` (NaN if no complete pairs).
    - `counts_12_events`: number of observed `from->mid` transitions
    - `counts_complete_pairs`: number of `from->mid` transitions that were followed by a `mid->to` later

    Returns the DataFrame (modified copy if `inplace=False`).
    """
    if not inplace:
        df = replicate_df.copy()
    else:
        df = replicate_df

    expl_cols = [c for c in df.columns if c.startswith(expl_prefix)]
    if not expl_cols:
        raise ValueError(f"no columns starting with '{expl_prefix}' found in replicate_df")

    def _turn_index(col: str):
        try:
            return int(col.rsplit("_", 1)[1])
        except Exception:
            return col

    expl_cols = sorted(expl_cols, key=_turn_index)

    avg_list = []
    never2 = []
    never3 = []
    counts_12 = []
    counts_pairs = []

    for _, row in df[expl_cols].iterrows():
        series = row.to_numpy(dtype=float)
        n = len(series)

        # find indices where a direct transition from `from_val -> mid_val` occurs
        pos_mid = []
        for i in range(n - 1):
            if series[i] == from_val and series[i + 1] == mid_val:
                pos_mid.append(i + 1)  # position where mid_val is first observed

        counts_12.append(len(pos_mid))

        if len(pos_mid) == 0:
            avg_list.append(np.nan)
            never2.append(True)
            never3.append(True)
            counts_pairs.append(0)
            continue

        never2.append(False)

        durations = []
        complete_pairs = 0
        for pos in pos_mid:
            # search for next transition mid->to after pos
            found = False
            for j in range(pos, n - 1):
                if series[j] == mid_val and series[j + 1] == to_val:
                    pos_to = j + 1
                    # number of turns between reaching mid and reaching to
                    durations.append(pos_to - pos)
                    complete_pairs += 1
                    found = True
                    break
            if not found:
                # censored event (mid reached but to not reached later)
                pass

        counts_pairs.append(complete_pairs)
        if complete_pairs == 0:
            avg_list.append(np.nan)
            never3.append(True)
        else:
            avg_list.append(float(np.mean(durations)))
            never3.append(False)

    df["never_reaches_2"] = never2
    df["never reaches 2"] = never2
    df["never_reaches_3"] = never3
    df["never reaches 3"] = never3
    df["avg_turns_between_12_23"] = avg_list
    df["counts_12_events"] = counts_12
    df["counts_complete_pairs"] = counts_pairs

    return df


def plot_avg_turns_between_increases(
    replicate_df: pd.DataFrame,
    expl_prefix: str = "exploration_count_",
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 4),
    title: Optional[str] = None,
    **compute_kwargs,
):
    """Compute per-row averages and plot a histogram of `avg_turns_between_12_23`.

    Returns the axes and the DataFrame produced by `compute_time_between_increases`.
    """
    df = compute_time_between_increases(replicate_df, expl_prefix=expl_prefix, **compute_kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = df["avg_turns_between_12_23"].dropna()
    ax.hist(data, bins=bins, color="#4C72B0", edgecolor="black")
    ax.set_xlabel("avg turns between 1->2 and subsequent 2->3")
    ax.set_ylabel("count (trajectories)")
    if title is None:
        ax.set_title("Histogram of average turns between 1->2 and 2->3")
    else:
        ax.set_title(title)

    # annotate with counts for never-reaching flags
    n_total = len(df)
    n_never2 = int(df["never_reaches_2"].sum()) if "never_reaches_2" in df.columns else 0
    n_never3 = int(df["never_reaches_3"].sum()) if "never_reaches_3" in df.columns else 0
    pct_never2 = 100.0 * n_never2 / n_total if n_total else 0.0
    pct_never3 = 100.0 * n_never3 / n_total if n_total else 0.0

    stats_txt = (
        f"n_total = {n_total}\n"
        f"never reaches 2: {n_never2} ({pct_never2:.1f}%)\n"
        f"never reaches 3: {n_never3} ({pct_never3:.1f}%)"
    )

    # place the annotation on the top-right inside the axes
    ax.text(
        0.97,
        0.97,
        stats_txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.8),
    )

    return ax, df


def filter_exclude_first_burst(
    replicate_df: pd.DataFrame,
    exclude_col: str = "exploration_count_1",
    exclude_value: float = 2.0,
) -> pd.DataFrame:
    """Return a copy of `replicate_df` excluding rows where
    `exclude_col == exclude_value`.

    Example usage:
        filtered = filter_exclude_first_burst(df)
        ax, out_df = plot_avg_turns_between_increases(filtered)
    """
    if exclude_col in replicate_df.columns:
        return replicate_df[replicate_df[exclude_col] != exclude_value].copy()
    return replicate_df.copy()


def plot_reach_proportions(
    replicate_df: pd.DataFrame,
    expl_prefix: str = "exploration_count_",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5, 4),
    title: Optional[str] = None,
):
    """Plot bar chart with proportions of replicates that:
    - reach 3 (eventually),
    - reach 2 (i.e. observed a 1->2 direct transition).

    Returns (ax, df) where `df` is the output of `compute_time_between_increases`.
    """
    df = compute_time_between_increases(replicate_df, expl_prefix=expl_prefix)

    n_total = len(df)
    if n_total == 0:
        raise ValueError("replicate_df is empty")

    # columns added by compute_time_between_increases: never_reaches_2, never_reaches_3
    never2 = df["never_reaches_2"] if "never_reaches_2" in df.columns else pd.Series([True] * n_total)
    never3 = df["never_reaches_3"] if "never_reaches_3" in df.columns else pd.Series([True] * n_total)

    reach2_count = int((~never2).sum())
    reach3_count = int((~never3).sum())

    categories = ["reach 3", "reach 2"]
    counts = [reach3_count, reach2_count]
    props = [100.0 * c / n_total for c in counts]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2ca02c", "#1f77b4"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="black")

    # annotate counts and percentages above bars
    for rect, cnt, pct in zip(bars, counts, props):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + max(1, n_total * 0.01),
            f"{cnt} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("count of trajectories")
    if title is None:
        ax.set_title("Reach proportions")
    else:
        ax.set_title(title)

    return ax, df
