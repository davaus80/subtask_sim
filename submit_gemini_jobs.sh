#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Usage: $0 <superfolder>"
    exit 1
fi

CONFIG_DIR="$1"
echo "Searching for subfolders in $CONFIG_DIR"

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
        for ((j=0;j<${#SUBFOLDERS[@]};j++)); do
            cancel_idx[$j]=1
        done
    else
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

for ((j=0;j<${#SUBFOLDERS[@]};j++)); do
    if [[ -n "${cancel_idx[$j]}" ]]; then
        echo "Skipping ${SUBFOLDERS[$j]}"
        continue
    fi
    echo "Submitting job for ${SUBFOLDERS[$j]}"
    sbatch ./run_slurm_gemini.sh "${SUBFOLDERS[$j]}"
done