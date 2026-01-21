#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------
# Path to your Overleaf (paper) repo
OVERLEAF_REPO="../icml2026_submission"

# Destination figures directory in Overleaf
OVERLEAF_FIG_DIR="$OVERLEAF_REPO/figures"

# Source figures directory (may contain arbitrarily deep subfolders)
SRC_FIG_DIR="figures"
# ------------------------

# Sanity checks
if [ ! -d "$SRC_FIG_DIR" ]; then
  echo "❌ Source figures directory '$SRC_FIG_DIR' does not exist."
  exit 1
fi

if [ ! -d "$OVERLEAF_REPO/.git" ]; then
  echo "❌ '$OVERLEAF_REPO' does not look like a git repo."
  exit 1
fi

mkdir -p "$OVERLEAF_FIG_DIR"

echo "🔄 Syncing nested figure tree → Overleaf repo..."

rsync -av --delete \
  --prune-empty-dirs \
  --exclude=".DS_Store" \
  "$SRC_FIG_DIR/" \
  "$OVERLEAF_FIG_DIR/"

echo "✅ All nested figure folders synced to:"
echo "   $OVERLEAF_FIG_DIR"
echo
echo "Next steps:"
echo "  cd $OVERLEAF_REPO"
echo "  git status"
echo "  git add figures/."
echo "  git commit -m \"Update figures\""
echo "  git push"
