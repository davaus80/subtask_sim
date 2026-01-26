#!/bin/bash

# Specify the directory containing subfolders
CONFIG_DIR="experiments/20260124_abandit_4sd_var/model_name_Qwen3-32B_prompt_summhist"
echo "Searching for subfolders in $CONFIG_DIR"
# Find all subfolders (non-recursively)
mapfile -t SUBFOLDERS < <(find "$CONFIG_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

echo "#### Found the following subfolders: ####"
i=1
for SUBFOLDER in "${SUBFOLDERS[@]}"; do
    echo "[$i] $SUBFOLDER"
    ((i++))
done

# Allow user to choose which subfolders to cancel (by number). Examples:
#  - Enter nothing (press Enter) to submit all
#  - Enter 'all' to cancel all (i.e. submit none)
#  - Enter comma-separated indices or ranges to cancel: 1,3,5-7
read -p "Enter indices (or ranges) of subfolders to CANCEL (e.g. 1,3,5-7). Press Enter to submit all: " cancel_input

cancel_input=${cancel_input,,}

declare -A cancel_idx=()

if [[ -n "$cancel_input" ]]; then
    if [[ "$cancel_input" == "all" ]]; then
        # mark all for cancellation
        for ((j=0;j<${#SUBFOLDERS[@]};j++)); do
            cancel_idx[$j]=1
        done
    else
        # parse comma separated and ranges
        IFS=',' read -ra parts <<< "$cancel_input"
        for p in "${parts[@]}"; do
            p=${p// /}
            if [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                start=${BASH_REMATCH[1]}
                end=${BASH_REMATCH[2]}
                if (( start <= end )); then
                    for ((k=start; k<=end; k++)); do
                        idx=$((k-1))
                        if (( idx>=0 && idx<${#SUBFOLDERS[@]} )); then
                            cancel_idx[$idx]=1
                        fi
                    done
                fi
            elif [[ "$p" =~ ^[0-9]+$ ]]; then
                idx=$((p-1))
                if (( idx>=0 && idx<${#SUBFOLDERS[@]} )); then
                    cancel_idx[$idx]=1
                fi
            fi
        done
    fi
fi

# Show what will be submitted vs cancelled
echo -e "\nSummary:"
to_submit=()
to_cancel=()
for ((j=0;j<${#SUBFOLDERS[@]};j++)); do
    if [[ -n "${cancel_idx[$j]}" ]]; then
        to_cancel+=("${SUBFOLDERS[$j]}")
    else
        to_submit+=("${SUBFOLDERS[$j]}")
    fi
done

echo "Will submit (${#to_submit[@]}) subfolders:"
for c in "${to_submit[@]}"; do echo "  $c"; done
echo "Will cancel (${#to_cancel[@]}) subfolders:"
for c in "${to_cancel[@]}"; do echo "  $c"; done

read -p "Proceed with submission of the ${#to_submit[@]} jobs? (y/n): " proceed
proceed=${proceed,,}
if [[ "$proceed" != "y" && "$proceed" != "yes" ]]; then
    echo "Aborting. No jobs submitted."
    exit 0
fi

# Submit a batch job for each subfolder
for ((j=0;j<${#SUBFOLDERS[@]};j++)); do
    if [[ -n "${cancel_idx[$j]}" ]]; then
        echo "Skipping ${SUBFOLDERS[$j]}"
        continue
    fi
    echo "Submitting job for ${SUBFOLDERS[$j]}"
    sbatch ./run_slurm_on_folder.sh "${SUBFOLDERS[$j]}"
done