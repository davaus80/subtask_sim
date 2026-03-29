.PHONY: sync-overleaf sync-overleaf-dry

# Two-way sync: push figures to Overleaf, pull .tex files back, commit+push
sync-overleaf:
	./sync_overleaf.sh

# Same, but skip the final git push to Overleaf
sync-overleaf-dry:
	./sync_overleaf.sh --no-push
