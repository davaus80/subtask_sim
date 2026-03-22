#!/usr/bin/env python3
"""
Replaces update_all_missing_logs.sh + find_incomplete_scalesweep_runs.sh.

A shuffle is considered complete if any .jsonl in its directory either:
  (a) has >= EXPECTED_ROW_COUNT lines, OR
  (b) contains >= 2 unique values for "action_taken_name"

Empty/zero-line .jsonl files are deleted (same behaviour as the bash version).

Output: missing_runs/<superfolder_name>/missing_runs.csv
"""

import json
import os
import sys
from pathlib import Path

FOLDERS = [
    # "experiments/20260120_ab_novar",
    # "experiments/20260124_abandit_2sd_var",
    # "experiments/20260124_abandit_4sd_var",
    # "experiments/20260118_farm_novar",
    # "experiments/20260124_farm_2sd_var",
    # "experiments/20260124_farm_4sd_var",
    # "experiments/20260312_rec_novar",
    # "experiments/20260312_rec_4sd_var",
    # "experiments/20260312_rec_2sd_var",
    # "experiments/20260311_gemini_farm_novar",
    # "experiments/20260312_gemini_ab_novar",
    "experiments/20260320_ab_scalesweep_novar",
    "experiments/20260320_farm_scalesweep_novar",
    "experiments/20260320_clothing_scalesweep_novar",
]

EXPECTED_SHUFFLE_COUNT = 20
EXPECTED_ROW_COUNT = 30


def model_size(model_dir_name: str) -> str:
    name = model_dir_name.lower()
    for tag in ("8b", "14b", "13b", "32b"):
        if tag in name:
            return tag.upper()
    if "gemini" in name:
        return "gemini"
    return ""


def is_shuffle_complete(shuffle_dir: Path) -> bool:
    for jsonl in shuffle_dir.glob("*.jsonl"):
        # Delete empty files
        if jsonl.stat().st_size == 0:
            print(f"Deleting empty file {jsonl}", file=sys.stderr)
            jsonl.unlink()
            continue
        try:
            lines = jsonl.read_text().splitlines()
        except OSError:
            continue
        lines = [l for l in lines if l.strip()]
        if not lines:
            print(f"Deleting zero-line file {jsonl}", file=sys.stderr)
            jsonl.unlink()
            continue
        if len(lines) >= EXPECTED_ROW_COUNT:
            return True
        unique_actions = set()
        for line in lines:
            try:
                unique_actions.add(json.loads(line).get("action_taken_name"))
            except json.JSONDecodeError:
                pass
        if len(unique_actions) >= 2:
            return True
    return False


def process_superfolder(superfolder: Path) -> list[tuple[str, int, str]]:
    missing = []
    logged_dirs = set()

    # Walk all shuffles/ directories
    for shuffles_dir in sorted(superfolder.rglob("shuffles")):
        if not shuffles_dir.is_dir():
            continue
        task_variation_dir = shuffles_dir.parent
        model_dir = task_variation_dir.parent
        size = model_size(model_dir.name)

        complete = sum(
            1 for d in sorted(shuffles_dir.iterdir())
            if d.is_dir() and is_shuffle_complete(d)
        )
        total_jsonl = sum(1 for _ in shuffles_dir.rglob("*.jsonl"))
        print(
            f"Task variation: {task_variation_dir}, "
            f"COMPLETE_COUNT={complete}, "
            f"EXPECTED_SHUFFLE_COUNT={EXPECTED_SHUFFLE_COUNT}, "
            f"TOTAL_JSONL_COUNT={total_jsonl}",
            file=sys.stderr,
        )

        if complete < EXPECTED_SHUFFLE_COUNT:
            missing.append((str(task_variation_dir), EXPECTED_SHUFFLE_COUNT - complete, size))
        logged_dirs.add(task_variation_dir.resolve())

    # Task-variation dirs with no shuffles/ at all (identified by config.yaml)
    for config in sorted(superfolder.rglob("config.yaml")):
        task_dir = config.parent
        if task_dir.resolve() == superfolder.resolve():
            continue
        if task_dir.resolve() in logged_dirs:
            continue
        if not (task_dir / "shuffles").is_dir():
            size = model_size(task_dir.parent.name)
            print(f"Logged missing shuffles for {task_dir} (no shuffles/ directory)", file=sys.stderr)
            missing.append((str(task_dir), EXPECTED_SHUFFLE_COUNT, size))
            logged_dirs.add(task_dir.resolve())

    return missing


def main():
    for folder_str in FOLDERS:
        superfolder = Path(folder_str)
        print(f"Processing {superfolder}")

        output_dir = Path("missing_runs") / superfolder.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "missing_runs.csv"

        missing = process_superfolder(superfolder)

        with open(output_file, "w") as f:
            f.write("folder_path,shuffles_missing,model_size\n")
            for path, count, size in missing:
                f.write(f"{path},{count},{size}\n")

        if missing:
            print(f"Missing runs logged to {output_file}")
        else:
            print("All runs are complete. No missing runs found.")


if __name__ == "__main__":
    main()
