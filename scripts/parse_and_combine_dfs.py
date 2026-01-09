"""Combine eval_results.csv files into a single dataframe of regrets.

For each located `eval_results.csv` this script will:
- parse the `subfolder` string (using the same lightweight parsing used by
  `scripts/parse_and_plot_barchart.py`),
- determine `dataset` from the file path (contains "abandit" -> 'abandit',
  otherwise 'farm'),
- extract a coarse `model_scale` label (e.g. '4B','8B','14B') from the
  folder path if present,
- collect all `cum_regret_{i}_mean` and `cum_regret_{i}_std` columns and
  re-name them to `regret@{i}_mean` / `regret@{i}_std`,
- aggregate across files by (`dataset`, `arm`, `scale`, `model_scale`) by
  taking the mean of available columns, and
- write the resulting combined CSV to disk.

Usage:
  python scripts/parse_and_combine_dfs.py --roots experiments/20251207_prompt_testing experiments/20251209_farm_sentiment experiments/20251209_abandit --out combined_regrets.csv
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse parsing helper(s) from the barchart script.
# Import robustly so the script works when run as:
#   python scripts/parse_and_combine_dfs.py
# and also when run as a module from the repo root.
try:
	from scripts.parse_and_plot_barchart import find_eval_csvs, parse_subfolder
except ModuleNotFoundError:
	# When running the file directly, the script directory is on sys.path,
	# so import the sibling module by name instead.
	from parse_and_plot_barchart import find_eval_csvs, parse_subfolder




def detect_dataset_from_path(path: str) -> str:
	p = path.lower()
	if "abandit" in p or "ab_bandit" in p:
		return "abandit"
	# fallback to 'farm' for other experiment folders
	return "farm"


def extract_model_scale_from_path(path: str) -> Optional[str]:
	"""Look for tokens like '4B', '8B', '14B' in the path segments.

	Returns one of '4B','8B','14B' if found, otherwise None.
	Prioritizes '14B' over '4B' to avoid substring misdetection.
	"""
	toks = re.split(r"[\/]+", path)
	# Check for 14B first to avoid matching '4B' in '14B'
	for t in toks[::-1]:
		# Check for 14B, 8B, 4B in order
		for scale in ("14B", "8B", "4B", "14b", "8b", "4b"):
			if scale in t:
				return scale.upper()
		# Fallback: regex for \d+[bB] (but check 14, 8, 4 in order)
		m = re.search(r"(14|8|4)[bB]\b", t)
		if m:
			return f"{m.group(1)}B"
	# Also try raw substring check in the whole path, prioritizing 14B
	for scale in ("14B", "8B", "4B", "14b", "8b", "4b"):
		if scale in path:
			return scale.upper()
	return None



def collect_and_combine(roots: List[str], out_file: str, subfolder_col: str = "subfolder") -> pd.DataFrame:
	files = find_eval_csvs(roots)
	if not files:
		raise FileNotFoundError(f"No eval_results.csv files found under: {roots}")

	rows = []
	for p in files:
		df = pd.read_csv(p)
		if subfolder_col not in df.columns:
			raise KeyError(f"Column '{subfolder_col}' not found in {p}. Available columns: {list(df.columns)}")
		parsed = df[subfolder_col].fillna("").astype(str).apply(lambda x: parse_subfolder(x))
		parsed_df = pd.json_normalize(parsed)
		df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
		df["_source_csv"] = os.path.basename(p)
		df["_source_path"] = p
		df["dataset"] = detect_dataset_from_path(p)
		df["model_scale"] = extract_model_scale_from_path(p) or "unknown"
		rows.append(df)

	full = pd.concat(rows, ignore_index=True, sort=False)

	# Find all cum_regret_{i}_(mean|std) columns and map to regret@{i}_mean/std
	pattern = re.compile(r"^cum_regret_(\d+)_(mean|std)$")
	found = {}
	for c in full.columns:
		m = pattern.match(c)
		if m:
			i = int(m.group(1))
			typ = m.group(2)
			found.setdefault(i, {})[typ] = c

	if not found:
		raise KeyError("No 'cum_regret_{i}_{mean|std}' columns found in the data files.")

	# Build columns for each i, using 'cum_regret@{i}_mean' and 'cum_regret@{i}_std' as output names
	all_is = sorted(found.keys())
	mean_cols = {}
	std_cols = {}
	for i in all_is:
		src_mean = found[i].get("mean")
		src_std = found[i].get("std")
		if src_mean:
			mean_cols[f"cum_regret@{i}_mean"] = src_mean
		if src_std:
			std_cols[f"cum_regret@{i}_std"] = src_std

	# Keep grouping keys and selected columns, now including 'prompt'
	keep_cols = ["dataset", "arm", "scale", "model_scale", "prompt"]
	selected = keep_cols + list(mean_cols.values()) + list(std_cols.values())

	# Subset (preserve rows where arm/scale exist)
	sub = full[selected].copy()

	# Rename source columns to target regret@ names for clarity before grouping
	rename_map = {v: k for k, v in {**mean_cols, **std_cols}.items()}
	sub = sub.rename(columns=rename_map)

	# Group by keys and take mean for all regret columns (mean of means / mean of stds)
	group_keys = ["dataset", "arm", "scale", "model_scale", "prompt"]
	agg = {c: "mean" for c in sub.columns if c not in group_keys}
	grouped = sub.groupby(group_keys, dropna=False).agg(agg).reset_index()


	# Reorder cum_regret columns by i
	regret_cols = []
	for i in all_is:
		mcol = f"cum_regret@{i}_mean"
		scol = f"cum_regret@{i}_std"
		if mcol in grouped.columns:
			regret_cols.append(mcol)
		if scol in grouped.columns:
			regret_cols.append(scol)

	out_cols = ["dataset", "arm", "scale", "model_scale", "prompt"] + regret_cols
	final = grouped.loc[:, [c for c in out_cols if c in grouped.columns]]

	# Ensure output directory exists
	odir = os.path.dirname(out_file) or "."
	os.makedirs(odir, exist_ok=True)
	final.to_csv(out_file, index=False)
	return final


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Combine eval_results.csv files into a single regrets dataframe CSV.")
	p.add_argument("--roots", required=True, nargs='+', help="One or more root folders to search for eval_results.csv files")
	p.add_argument("--out", default="plots/combined_regrets.csv", help="Output CSV path")
	return p


if __name__ == "__main__":
	parser = _build_parser()
	args = parser.parse_args()
	print(f"Searching roots: {args.roots}")
	out = collect_and_combine(args.roots, args.out)
	print(f"Wrote combined dataframe to: {args.out}")
	print(out.head().to_string())

