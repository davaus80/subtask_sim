#!/bin/bash

# Usage: ./submit_calibrated_scalesweep_jobs.sh path/to/missing_runs.csv
#
# Submits 5 shuffles per job for scalesweep experiments.
# Time is calibrated using:
#   base a=15 min/shuffle for 32B
#   multipliers:
#     negative scale:            x0.4
#     alphanumeric + pos scale:  x0.5
#     other (helpful/mislead, pos scale): x1.0

SHUFFLES_PER_JOB=5
gemini_count=0
gemini_delay=0

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path/to/missing_runs.csv"
    exit 1
fi

MISSING_RUNS_FILE=$1

if [ ! -f "$MISSING_RUNS_FILE" ]; then
    echo "Error: File $MISSING_RUNS_FILE does not exist."
    exit 1
fi

while IFS=, read -r folder_path shuffles_missing model_size || [ -n "$folder_path" ]; do
    if [ "$folder_path" == "folder_path" ]; then
        continue
    fi

    if [[ "$model_size" == "gemini" ]]; then
        if (( gemini_count > 0 && gemini_count % 10 == 0 )); then
            gemini_delay=$(( gemini_delay + 20 ))
        fi
        echo "Submitting gemini job for $folder_path (begin=now+${gemini_delay}min)"
        sbatch --time=00:15:00 --begin=now+${gemini_delay}minutes ./run_slurm_gemini.sh "$folder_path"
        gemini_count=$(( gemini_count + 1 ))
        continue
    fi

    # Parse scale sign and nomenclature prefix from final path component
    folder_name=$(basename "$folder_path")
    scale_value=$(echo "$folder_name" | grep -oP '(?<=_scale)-?[0-9.]+')
    is_negative=false
    is_alphanumeric=false

    if echo "$scale_value" | grep -qP '^-'; then
        is_negative=true
    fi

    if echo "$folder_name" | grep -qP '^alphanumeric_'; then
        is_alphanumeric=true
    fi

    # Base coefficient (min/shuffle) per model size
    case "$model_size" in
        "32B")
            a=15
            gpu_gres="gpu:rtx8000:2"
            ;;
        "14B")
            a=18
            gpu_gres="gpu:rtx8000:2"
            ;;
        "8B")
            a=10
            gpu_gres="gpu:rtx8000:1"
            ;;
        *)
            echo "Unknown model size: $model_size. Skipping."
            continue
            ;;
    esac

    b=5

    # Apply multiplier (scaled to integer arithmetic via tenths)
    # negative scale: x0.4 → multiply a by 4, divide by 10
    # alphanumeric + positive: x0.5 → multiply a by 5, divide by 10
    # other: x1.0
    if $is_negative; then
        time_minutes=$(( (a * 4 * SHUFFLES_PER_JOB) / 10 + b ))
    elif $is_alphanumeric; then
        time_minutes=$(( (a * 5 * SHUFFLES_PER_JOB) / 10 + b ))
    else
        time_minutes=$(( a * SHUFFLES_PER_JOB + b ))
    fi

    hours=$((time_minutes / 60))
    minutes=$((time_minutes % 60))
    time_formatted=$(printf "%02d:%02d:00" $hours $minutes)

    echo "Submitting job for $folder_path | time=$time_formatted | GPUs=$gpu_gres | scale=$scale_value | alphanumeric=$is_alphanumeric | negative=$is_negative"

    sbatch --time=$time_formatted --gres=$gpu_gres ./run_slurm_on_folder.sh "$folder_path" "$SHUFFLES_PER_JOB"

done < "$MISSING_RUNS_FILE"
