#!/usr/bin/env zsh
# Update all eval results, figures, and sync to Overleaf.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# 1. Re-run all evals
echo "==> Running eval_all.sh"
./eval_all.sh

# 2. Re-run viz notebooks
NOTEBOOKS=(
    src/viz/data_processing.ipynb
    src/viz/appendix_tables.ipynb
    src/viz/label_channel_bias.ipynb
    src/viz/reward_channel_bias.ipynb
    src/viz/scale_sweep.ipynb
)

for nb in "${NOTEBOOKS[@]}"; do
    echo "==> Executing $nb"
    jupyter nbconvert --to notebook --inplace --execute "$nb"
done

# 3. Sync figures and .tex to Overleaf
echo "==> Syncing to Overleaf"
./sync_overleaf.sh
