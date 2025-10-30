#!/bin/bash

# OPTION 1: SUBMIT ALL YAMLS INSIDE A FOLDER, RECURSIVELY
CONFIG_DIR="experiments/basic_novar/name_shuffles/semantic_error"
echo "Searching for configs in $CONFIG_DIR"
# Find all YAML files recursively
mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -type f -name "*.yaml" | sort)

# OPTION 2: CUSTOM SELECTION OF CONFIGS
# CONFIGS_FILES=("config1.yaml" "config2.yaml" "config3.yaml")


echo "#### Found the following configs: ####"
i=1
for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "[$i] $CONFIG"
    ((i++))
done

# Allow user to choose which configs to cancel (by number). Examples:
#  - Enter nothing (press Enter) to submit all
#  - Enter 'all' to cancel all (i.e. submit none)
#  - Enter comma-separated indices or ranges to cancel: 1,3,5-7
read -p "Enter indices (or ranges) of configs to CANCEL (e.g. 1,3,5-7). Press Enter to submit all: " cancel_input

cancel_input=${cancel_input,,}

declare -A cancel_idx=()

if [[ -n "$cancel_input" ]]; then
    if [[ "$cancel_input" == "all" ]]; then
        # mark all for cancellation
        for ((j=0;j<${#CONFIG_FILES[@]};j++)); do
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
                        if (( idx>=0 && idx<${#CONFIG_FILES[@]} )); then
                            cancel_idx[$idx]=1
                        fi
                    done
                fi
            elif [[ "$p" =~ ^[0-9]+$ ]]; then
                idx=$((p-1))
                if (( idx>=0 && idx<${#CONFIG_FILES[@]} )); then
                    cancel_idx[$idx]=1
                fi
            fi
        done
    fi
fi

# Show what will be submitted vs cancelled
echo "\nSummary:"
to_submit=()
to_cancel=()
for ((j=0;j<${#CONFIG_FILES[@]};j++)); do
    if [[ -n "${cancel_idx[$j]}" ]]; then
        to_cancel+=("${CONFIG_FILES[$j]}")
    else
        to_submit+=("${CONFIG_FILES[$j]}")
    fi
done

echo "Will submit (${#to_submit[@]}) configs:"
for c in "${to_submit[@]}"; do echo "  $c"; done
echo "Will cancel (${#to_cancel[@]}) configs:"
for c in "${to_cancel[@]}"; do echo "  $c"; done

read -p "Proceed with submission of the ${#to_submit[@]} jobs? (y/n): " proceed
proceed=${proceed,,}
if [[ "$proceed" != "y" && "$proceed" != "yes" ]]; then
    echo "Aborting. No jobs submitted."
    exit 0
fi

# Check if user is ok with the configs selected
read -p "Continue? (y/n): " answer

# Normalize input to lowercase (optional but nice)
answer=${answer,,}

if [[ "$answer" == "y" || "$answer" == "yes" ]]; then
    echo "Continuing..."
    # your code here
else
    echo "Exiting."
    exit 0
fi

# Submit a batch job for each config in CONFIG_FILES
for ((j=0;j<${#CONFIG_FILES[@]};j++)); do
    if [[ -n "${cancel_idx[$j]}" ]]; then
        echo "Skipping ${CONFIG_FILES[$j]}"
        continue
    fi
    echo "Submitting job for ${CONFIG_FILES[$j]}"
    sbatch run_slurm_job.sh "${CONFIG_FILES[$j]}"
done