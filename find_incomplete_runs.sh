#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <path_to_experiment_group_name_folder> [expected_shuffle_count] [expected_row_count]"
  exit 1
fi

# Parse arguments
EXPERIMENT_GROUP_PATH=$1
EXPECTED_SHUFFLE_COUNT=${2:-10}
EXPECTED_ROW_COUNT=${3:-10}

# Define output directory and file
OUTPUT_DIR="./missing_runs/$(basename "$EXPERIMENT_GROUP_PATH")"
OUTPUT_FILE="$OUTPUT_DIR/missing_runs.csv"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize the output file
> "$OUTPUT_FILE"

# Write CSV header to the output file
echo "folder_path,shuffles_missing,model_size" > "$OUTPUT_FILE"

# Traverse the subdirectory tree and find incomplete runs
find "$EXPERIMENT_GROUP_PATH" -type d -name "shuffles" | while read -r SHUFFLE_DIR; do
  TASK_VARIATION_DIR=$(dirname "$SHUFFLE_DIR")
  PARAM_CONFIG=$(basename $(dirname "$TASK_VARIATION_DIR"))

  # Extract model size from param_configuration
  MODEL_SIZE=""
  if [[ "$PARAM_CONFIG" == *"8B"* ]]; then
    MODEL_SIZE="8B"
  elif [[ "$PARAM_CONFIG" == *"14B"* ]]; then
    MODEL_SIZE="14B"
  elif [[ "$PARAM_CONFIG" == *"32B"* ]]; then
    MODEL_SIZE="32B"
  fi

  COMPLETE_COUNT=0

  # Ensure the path to jsonl files is correct
  for SHUFFLE_SUBDIR in "$SHUFFLE_DIR"/*; do
    if [ -d "$SHUFFLE_SUBDIR" ]; then
      JSONL_FILE=$(find "$SHUFFLE_SUBDIR" -type f -name "run_*.jsonl" | head -n 1)
    #   echo "Checking for JSONL file: $JSONL_FILE" >&2
      if [ -n "$JSONL_FILE" ]; then
        ROW_COUNT=$(wc -l < "$JSONL_FILE")
        if [ "$ROW_COUNT" -ge "$EXPECTED_ROW_COUNT" ]; then
          COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
        fi
      fi
    fi
  done

  # Debugging output to count all jsonl files in the shuffle folder
  TOTAL_JSONL_COUNT=$(find "$SHUFFLE_DIR" -type f -name "results.jsonl" | wc -l)
#   echo "Task variation: $TASK_VARIATION_DIR, TOTAL_JSONL_COUNT=$TOTAL_JSONL_COUNT, COMPLETE_COUNT=$COMPLETE_COUNT, EXPECTED_SHUFFLE_COUNT=$EXPECTED_SHUFFLE_COUNT" >&2

  # Debugging output to verify row count and shuffle completeness
  for JSONL_FILE in "$SHUFFLE_DIR"/shuffle_*; do
    if [ -f "$JSONL_FILE" ]; then
      ROW_COUNT=$(wc -l < "$JSONL_FILE")
    #   echo "Checking $JSONL_FILE: ROW_COUNT=$ROW_COUNT, EXPECTED_ROW_COUNT=$EXPECTED_ROW_COUNT" >&2
    fi
  done

  # Debugging output to verify task variation completeness
  echo "Task variation: $TASK_VARIATION_DIR, COMPLETE_COUNT=$COMPLETE_COUNT, EXPECTED_SHUFFLE_COUNT=$EXPECTED_SHUFFLE_COUNT" >&2

  # Count the number of missing shuffle folders
  MISSING_SHUFFLES=$((EXPECTED_SHUFFLE_COUNT - COMPLETE_COUNT))

  # If the number of complete results is less than expected, log the task variation folder and missing count
  if [ "$COMPLETE_COUNT" -lt "$EXPECTED_SHUFFLE_COUNT" ]; then
    echo "$TASK_VARIATION_DIR,$MISSING_SHUFFLES,$MODEL_SIZE" >> "$OUTPUT_FILE"
  fi

done

# Notify the user
if [ -s "$OUTPUT_FILE" ]; then
  echo "Missing runs logged to $OUTPUT_FILE"
else
  echo "All runs are complete. No missing runs found."
fi
