"""
run_baselines.py

Runs Thompson Sampling and UCB1 baselines for positive and negative reward
conditions, then writes eval_results.csv files in the format expected by
eval_manager.py and data_processing.ipynb.

Output structure
----------------
experiments/20260326_classicalsolver/
  UCB1/
    high_scale/         eval_results.csv   (3 rows: sigma = 0, 6.25, 12.5)
    high_neg_scale/     eval_results.csv
  TS/
    high_scale/         eval_results.csv
    high_neg_scale/     eval_results.csv

Each CSV row corresponds to one noise level (sigma). Columns match the
eval_manager.py output format so the same plotting/processing code applies:
  nomenclature, scale, variance, sigma, n_shuffles,
  cum_regret_{t}_mean, cum_regret_{t}_std,          (t = 0 .. T-1)
  exploration_count_{t}_mean, exploration_count_{t}_std

Usage
-----
    python run_baselines.py
"""

import os
import numpy as np
import pandas as pd

from src.baselines import ThompsonSampling, UCB1, run_baseline

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

T = 10        # episode length (turns)
N = 10000      # replicates per condition
SEED = 42

# Reward conditions: (folder_name, arm_means)
REWARD_CONDITIONS = {
    "high_scale":     [75.0, 50.0, 25.0],
    "high_neg_scale": [-25.0, -50.0, -75.0],
}

# Noise levels: variance label (matches eval_manager convention) -> sigma value
# No = deterministic, Low = 4-sigma gap (sigma = gap/4 = 25/4 = 6.25),
# High = 2-sigma gap (sigma = gap/2 = 25/2 = 12.5)
SIGMA_CONDITIONS = {
    "No":   0.0,
    "Low":  6.25,
    "High": 12.5,
}

BASE_DIR = "experiments/20260326_classicalsolver"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(method: str, sigma: float):
    if method == "UCB1":
        return UCB1(k=3)
    elif method == "TS":
        return ThompsonSampling(k=3, sigma=sigma)
    raise ValueError(f"Unknown method: {method}")


def run_condition(method: str, scale_name: str, means: list) -> None:
    out_dir = os.path.join(BASE_DIR, method, scale_name)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for var_name, sigma in SIGMA_CONDITIONS.items():
        agent = make_agent(method, sigma)
        results = run_baseline(agent, means, sigma, T, N, seed=SEED)

        row: dict = {
            "nomenclature": "baseline",
            "scale": scale_name,
            "variance": var_name,
            "sigma": sigma,
            "n_shuffles": N,
        }
        for t in range(T):
            row[f"cum_regret_{t}_mean"]          = float(np.mean(results["cum_regrets"][:, t]))
            row[f"cum_regret_{t}_std"]           = float(np.std(results["cum_regrets"][:, t], ddof=0))
            row[f"exploration_count_{t}_mean"]   = float(np.mean(results["exploration_counts"][:, t]))
            row[f"exploration_count_{t}_std"]    = float(np.std(results["exploration_counts"][:, t], ddof=0))

        rows.append(row)

    csv_path = os.path.join(out_dir, "eval_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for method in ["UCB1", "TS"]:
        for scale_name, means in REWARD_CONDITIONS.items():
            run_condition(method, scale_name, means)
