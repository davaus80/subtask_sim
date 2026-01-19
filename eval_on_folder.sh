#!/bin/zsh
PARENT_DIR=$1
for dir in $PARENT_DIR/*(/); do
  python ./eval_manager.py --superfolder_path "$dir" --min_rows 10 --output eval_results.csv
done