from typing import Optional, List, Dict
"""Evaluation manager for experiment results.

Traverses a "superfolder" containing subfolders (one row per subfolder in the final
CSV). Each subfolder contains multiple shuffle subfolders; in each shuffle we
select the most recent .jsonl that contains at least `min_rows` records. We then
compute simple per-shuffle metrics (total reward and mean per-step reward) and
aggregate (mean/std) across shuffles for each subfolder.

Usage:
    python eval_manager.py --superfolder_path experiments/20251104_fb --min_rows 10

The resulting CSV is written to <superfolder>/eval_results.csv by default.
"""
from typing import Optional, List, Dict
import logging
import os
import sys
from argparse import ArgumentParser

import jsonlines
import pandas as pd
import numpy as np
import yaml


logging.basicConfig(level=logging.INFO)


def select_results_file_from_dir(dir_path: str, min_rows: int = 10) -> Optional[str]:
    """Return the most-recent .jsonl in `dir_path` that contains at least `min_rows` records.

    If none found, returns None.
    """
    if not os.path.isdir(dir_path):
        logging.debug("select_results_file_from_dir: not a directory: %s", dir_path)
        return None

    # list jsonl files
    candidates = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jsonl')]
    if not candidates:
        return None

    # sort by modification time (most recent first)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for p in candidates:
        try:
            with jsonlines.open(p) as reader:
                count = 0
                for _ in reader:
                    count += 1
                    if count >= min_rows:
                        break
        except Exception as e:
            logging.warning("Failed to read %s: %s", p, e)
            continue

        if count >= min_rows:
            return p

    return None


def load_results_from_file(jsonl_path: str) -> List[Dict]:
    """Load a .jsonl file and return a list of dicts (one per step)."""
    results = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            results.append(obj)
    return results


def eval_loop(super_folder_path: str, min_rows: int = 10, output_filename: str = 'eval_results.csv'):
    """Traverse immediate subfolders of `super_folder_path`, find recent jsonl in each shuffle folder,
    compute simple metrics and write a CSV summary to `output_filename` inside the superfolder.

    Metrics currently computed per-subfolder (averaged across shuffles):
    - total_reward_mean / std (sum of 'reward' over first `min_rows` steps)
    - mean_reward_mean / std (mean per-step reward over first `min_rows` steps)
    """
    full_results: List[Dict] = []

    if not os.path.isdir(super_folder_path):
        raise ValueError(f"super_folder_path is not a directory: {super_folder_path}")

    subfolders = [os.path.join(super_folder_path, d) for d in os.listdir(super_folder_path) if os.path.isdir(os.path.join(super_folder_path, d))]
    subfolders.sort()

    for subpath in subfolders:
        subname = os.path.basename(subpath)
        # look for shuffle subfolders inside
        shuffle_dirs = [os.path.join(subpath, d) for d in os.listdir(subpath) if os.path.isdir(os.path.join(subpath, d))]
        if not shuffle_dirs:
            logging.info("No shuffle dirs found in %s -- skipping", subpath)
            continue

        per_shuffle_totals = []
        per_shuffle_means = []
        valid_shuffles = 0

        # Collect per-shuffle per-step metrics into lists so we can aggregate across shuffles
        # Each element will be a list of length min_rows
        per_shuffle_cum_rewards = []
        per_shuffle_step_rewards = []
        per_shuffle_regrets = []
        per_shuffle_optimal_props = []

        for sdir in shuffle_dirs:
            candidate = select_results_file_from_dir(sdir, min_rows=min_rows)
            if candidate is None:
                logging.warning("No valid jsonl with >=%d rows in %s", min_rows, sdir)
                continue

            try:
                results = load_results_from_file(candidate)
            except Exception as e:
                logging.warning("Failed to load results from %s: %s", candidate, e)
                continue

            if len(results) < min_rows:
                logging.warning("%s has %d rows (< %d) after loading; skipping", candidate, len(results), min_rows)
                continue

            # Trim to exactly min_rows to be consistent
            results_trim = results[:min_rows]

            rewards = [float(r.get('reward', 0.0)) for r in results_trim]
            # cumulative rewards per step
            cum_rewards = list(np.cumsum(rewards))
            per_shuffle_cum_rewards.append(cum_rewards)
            per_shuffle_step_rewards.append(rewards)

            # load config.yaml in this shuffle dir to determine optimal arm mean
            config_path = os.path.join(sdir, 'config.yaml')
            optimal_mean = None
            optimal_id = None
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as fh:
                        cfg = yaml.safe_load(fh)
                    # Expect config to have subtasks -> list[0] -> params -> actions
                    subtasks = cfg.get('subtasks') if isinstance(cfg, dict) else None
                    if subtasks and isinstance(subtasks, list) and len(subtasks) > 0:
                        actions_cfg = subtasks[0].get('params', {}).get('actions', [])
                        if actions_cfg:
                            # actions_cfg items have 'id' and 'mean'
                            best = max(actions_cfg, key=lambda a: float(a.get('mean', float('-inf'))))
                            optimal_mean = float(best.get('mean'))
                            optimal_id = best.get('id')
                except Exception:
                    logging.warning("Failed to load config.yaml in %s; skipping regret/optimal calculations", sdir)

            # compute regrets and optimal-action counts if we have optimal_mean
            if optimal_mean is not None:
                regrets = [optimal_mean - rw for rw in rewards]
                per_shuffle_regrets.append(regrets)
                optimal_counts = sum(1 for rdict in results_trim if rdict.get('action_taken_id') == optimal_id)
                per_shuffle_optimal_props.append(optimal_counts / float(min_rows))
            else:
                # mark with None so lengths match when aggregating
                per_shuffle_regrets.append([np.nan] * min_rows)
                per_shuffle_optimal_props.append(np.nan)

            valid_shuffles += 1

        if valid_shuffles == 0:
            logging.warning("No valid shuffles for %s; skipping subfolder.", subname)
            continue

        # Aggregate per-step metrics across shuffles
        cum_arr = np.array(per_shuffle_cum_rewards, dtype=float)  # shape (n_shuffles, min_rows)
        step_arr = np.array(per_shuffle_step_rewards, dtype=float)
        regret_arr = np.array(per_shuffle_regrets, dtype=float)

        row = {'subfolder': subname, 'n_shuffles': valid_shuffles}

        # cumulative and step-wise reward stats
        for t in range(min_rows):
            row[f'total_reward_{t}_mean'] = float(np.mean(cum_arr[:, t]))
            row[f'total_reward_{t}_std'] = float(np.std(cum_arr[:, t], ddof=0))
            row[f'reward_{t}_mean'] = float(np.mean(step_arr[:, t]))
            row[f'reward_{t}_std'] = float(np.std(step_arr[:, t], ddof=0))

            # regrets may contain nan if optimal not available for some shuffles
            row[f'regret_{t}_mean'] = float(np.nanmean(regret_arr[:, t]))
            row[f'regret_{t}_std'] = float(np.nanstd(regret_arr[:, t], ddof=0))

        # percent optimal action overall (mean/std across shuffles)
        prop_arr = np.array(per_shuffle_optimal_props, dtype=float)
        row['percent_optimal_mean'] = float(np.nanmean(prop_arr))
        row['percent_optimal_std'] = float(np.nanstd(prop_arr, ddof=0))

        full_results.append(row)

    if not full_results:
        logging.warning("No results collected from %s", super_folder_path)
        return None

    df = pd.DataFrame(full_results)
    out_path = os.path.join(super_folder_path, output_filename)
    df.to_csv(out_path, index=False)
    logging.info("Wrote eval results to %s", out_path)
    return df


def get_args(args):
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--superfolder_path", default=None, help="Path to the superfolder containing subfolders to evaluate")
    parser.add_argument("--min_rows", type=int, default=10, help="Minimum number of rows (steps) required in each selected jsonl (default: 10)")
    parser.add_argument("--output", default='eval_results.csv', help="Output CSV filename (written inside superfolder)")
    parser.add_argument("--help", action='store_true', help="Show help")
    return parser.parse_args(args)


if __name__ == "__main__":

    args = get_args(sys.argv[1:])

    if args.help or args.superfolder_path is None:
        print("Call signature: python ./eval_manager.py --superfolder_path <path> [--min_rows 10] [--output eval_results.csv]")
    else:
        df = eval_loop(args.superfolder_path, min_rows=args.min_rows, output_filename=args.output)
        if df is not None:
            print(f"Wrote summary csv to: {os.path.join(args.superfolder_path, args.output)}")

