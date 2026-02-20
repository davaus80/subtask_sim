#!/bin/bash

# Usage: ./submit_calibrated_jobs.sh path/to/missing_runs.csv

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path/to/missing_runs.csv"
    exit 1
fi

MISSING_RUNS_FILE=$1

if [ ! -f "$MISSING_RUNS_FILE" ]; then
    echo "Error: File $MISSING_RUNS_FILE does not exist."
    exit 1
fi

# Read the CSV file and process each line
while IFS=, read -r folder_path shuffles_missing model_size || [ -n "$folder_path" ]; do
    # Skip the header line
    if [ "$folder_path" == "folder_path" ]; then
        continue
    fi

    # Determine the coefficients based on model size
    case "$model_size" in
        "32B")
            a=6
            ;;
        "14B")
            a=4
            ;;
        "8B")
            a=2
            ;;
        *)
            echo "Unknown model size: $model_size. Skipping."
            continue
            ;;
    esac

    b=4

    # Calculate the time in minutes
    time_minutes=$((a * shuffles_missing + b))

    # Convert time to HH:MM:SS format
    hours=$((time_minutes / 60))
    minutes=$((time_minutes % 60))
    time_formatted=$(printf "%02d:%02d:00" $hours $minutes)

    echo "Submitting job for $folder_path with time $time_formatted"

    # Submit the job
    sbatch --time=$time_formatted ./run_slurm_on_folder.sh "$folder_path"
done < "$MISSING_RUNS_FILE"