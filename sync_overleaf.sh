#!/usr/bin/env bash
# sync_overleaf.sh — two-way sync between this repo and the Overleaf git repo.
#
# PUSH: copies figures listed in .overleaf_figures → <OVERLEAF_REPO>/figures/
# PULL: copies *.tex files from <OVERLEAF_REPO>/ → paper/
# Then commits and pushes to Overleaf.
#
# Usage:
#   ./sync_overleaf.sh           # full two-way sync + push to Overleaf
#   ./sync_overleaf.sh --no-push # sync locally, skip git push to Overleaf
set -euo pipefail

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OVERLEAF_REMOTE="https://github.com/davaus80/COLM-2026-Semantic-Biases"
OVERLEAF_REPO="../COLM-2026-Semantic-Biases"   # cloned as sibling folder
OVERLEAF_FIG_DIR="$OVERLEAF_REPO/figures"       # destination for figures inside Overleaf repo
OVERLEAF_TEX_DIR="$OVERLEAF_REPO"              # pull .tex from here
LOCAL_TEX_DIR="paper"                           # where pulled .tex files land locally
# Path mirroring: strip the first directory component of each source path.
# e.g. figures/main/foo.pdf  → main/foo.pdf → <overleaf>/figures/main/foo.pdf
# e.g. figures_png/regret/x/figure.png → regret/x/figure.png → <overleaf>/figures/regret/x/figure.png
FIGURES_CONFIG=".overleaf_figures"
COMMIT_MSG="Update figures from subtask_sim"
# ─────────────────────────────────────────────────────────────────────────────

PUSH=true
for arg in "$@"; do
  [[ "$arg" == "--no-push" ]] && PUSH=false
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# ── 1. Ensure Overleaf repo exists ───────────────────────────────────────────
if [ ! -d "$OVERLEAF_REPO/.git" ]; then
  echo "Cloning Overleaf repo → $OVERLEAF_REPO"
  git clone "$OVERLEAF_REMOTE" "$OVERLEAF_REPO"
else
  echo "Pulling latest from Overleaf..."
  git -C "$OVERLEAF_REPO" pull --rebase
fi

# ── 2. PULL: copy .tex files from Overleaf → paper/ ─────────────────────────
echo ""
echo "Pulling .tex files: $OVERLEAF_TEX_DIR → $LOCAL_TEX_DIR/"
mkdir -p "$LOCAL_TEX_DIR"
TEX_COUNT=0
while IFS= read -r -d '' texfile; do
  relpath="${texfile#$OVERLEAF_TEX_DIR/}"
  destfile="$LOCAL_TEX_DIR/$relpath"
  mkdir -p "$(dirname "$destfile")"
  cp "$texfile" "$destfile"
  echo "  pulled: $relpath"
  (( TEX_COUNT++ )) || true
done < <(find "$OVERLEAF_TEX_DIR" -name "*.tex" -not -path "*/.git/*" -print0)
echo "  $TEX_COUNT .tex file(s) copied to $LOCAL_TEX_DIR/"

# ── 3. PUSH: copy selected figures → Overleaf repo ──────────────────────────
if [ ! -f "$FIGURES_CONFIG" ]; then
  echo ""
  echo "Warning: $FIGURES_CONFIG not found — skipping figure push."
else
  echo ""
  echo "Pushing figures → $OVERLEAF_FIG_DIR/"
  mkdir -p "$OVERLEAF_FIG_DIR"
  FIG_COUNT=0
  while IFS= read -r line; do
    # strip comments and blank lines
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -z "$line" ]] && continue

    # expand globs — use find for ** patterns (bash 3.2 on macOS lacks globstar),
    # fall back to plain glob for simple patterns
    matches=()
    if [[ "$line" == *"**"* ]]; then
      base="${line%%\*\**}"
      base="${base%/}"
      name_pat="${line##*\*\*/}"
      [[ -z "$base" ]] && base="."
      while IFS= read -r f; do matches+=("$f"); done < <(find "$base" -name "$name_pat" -type f 2>/dev/null)
    else
      shopt -s nullglob
      matches=( $line )
      shopt -u nullglob
    fi

    if [ ${#matches[@]} -eq 0 ]; then
      echo "  Warning: no match for pattern '$line'"
      continue
    fi

    for src in "${matches[@]}"; do
      # strip first path component: figures/main/foo.pdf → main/foo.pdf
      relpath="${src#*/}"
      dest="$OVERLEAF_FIG_DIR/$relpath"
      mkdir -p "$(dirname "$dest")"
      cp "$src" "$dest"
      echo "  pushed: $relpath"
      (( FIG_COUNT++ )) || true
    done
  done < "$FIGURES_CONFIG"
  echo "  $FIG_COUNT figure(s) copied to $OVERLEAF_FIG_DIR/"
fi

# ── 4. Commit and optionally push to Overleaf ────────────────────────────────
echo ""
cd "$OVERLEAF_REPO"
if [ -z "$(git status --porcelain)" ]; then
  echo "No changes in Overleaf repo — nothing to commit."
else
  git add -A
  git commit -m "$COMMIT_MSG"
  echo "Committed changes to Overleaf repo."
fi

if $PUSH; then
  git push
  echo "Pushed to Overleaf remote."
else
  echo "(Skipped push — run 'git push' inside $OVERLEAF_REPO when ready.)"
fi

echo ""
echo "Done."
