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
      SHUFFLE_COMPLETE=0
      # Check every jsonl file in the shuffle subdirectory. If any file has
      # at least EXPECTED_ROW_COUNT rows, consider this shuffle complete.
      while IFS= read -r JSONL_FILE; do
        # If file is zero bytes or missing, remove it
        if [ ! -s "$JSONL_FILE" ]; then
          echo "Deleting empty file $JSONL_FILE" >&2
          rm -f "$JSONL_FILE"
          continue
        fi
        ROW_COUNT=$(wc -l < "$JSONL_FILE")
        if [ "$ROW_COUNT" -eq 0 ]; then
          echo "Deleting zero-line file $JSONL_FILE" >&2
          rm -f "$JSONL_FILE"
          continue
        fi
        if [ "$ROW_COUNT" -ge "$EXPECTED_ROW_COUNT" ]; then
          SHUFFLE_COMPLETE=1
          break
        fi
      done < <(find "$SHUFFLE_SUBDIR" -type f -name "*.jsonl")

      if [ "$SHUFFLE_COMPLETE" -eq 1 ]; then
        COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
      fi
    fi
  done

# Debugging output to count all jsonl files in the shuffle folder
  TOTAL_JSONL_COUNT=$(find "$SHUFFLE_DIR" -type f -name "*.jsonl" | wc -l)
#   echo "Task variation: $TASK_VARIATION_DIR, TOTAL_JSONL_COUNT=$TOTAL_JSONL_COUNT, COMPLETE_COUNT=$COMPLETE_COUNT, EXPECTED_SHUFFLE_COUNT=$EXPECTED_SHUFFLE_COUNT" >&2

  # Debugging output to verify task variation completeness
  echo "Task variation: $TASK_VARIATION_DIR, COMPLETE_COUNT=$COMPLETE_COUNT, EXPECTED_SHUFFLE_COUNT=$EXPECTED_SHUFFLE_COUNT, TOTAL_JSONL_COUNT=$TOTAL_JSONL_COUNT" >&2

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

# Additionally, include any task-variation folders that do not have a `shuffles/`
# subdirectory at all. We identify task-variation folders by looking for
# `config.yaml` files and taking their parent directory. If a task-variation
# folder lacks `shuffles/` then it is missing all expected shuffles.
while IFS= read -r TASK_DIR; do
  # Skip the experiment group root itself (don't log the top-level folder)
  if command -v realpath >/dev/null 2>&1; then
    if [ "$(realpath -s "$TASK_DIR")" = "$(realpath -s "$EXPERIMENT_GROUP_PATH")" ]; then
      continue
    fi
  else
    if [ "$TASK_DIR" = "$EXPERIMENT_GROUP_PATH" ]; then
      continue
    fi
  fi

  # Skip if the task variation already got logged
  if grep -qF "${TASK_DIR}," "$OUTPUT_FILE"; then
    continue
  fi

  if [ ! -d "$TASK_DIR/shuffles" ]; then
    PARAM_CONFIG=$(basename $(dirname "$TASK_DIR"))
    MODEL_SIZE=""
    if [[ "$PARAM_CONFIG" == *"8B"* ]]; then
      MODEL_SIZE="8B"
    elif [[ "$PARAM_CONFIG" == *"14B"* ]]; then
      MODEL_SIZE="14B"
    elif [[ "$PARAM_CONFIG" == *"32B"* ]]; then
      MODEL_SIZE="32B"
    fi

    MISSING_SHUFFLES=$EXPECTED_SHUFFLE_COUNT
    echo "${TASK_DIR},${MISSING_SHUFFLES},${MODEL_SIZE}" >> "$OUTPUT_FILE"
    echo "Logged missing shuffles for $TASK_DIR (no shuffles/ directory)" >&2
  fi
done < <(find "$EXPERIMENT_GROUP_PATH" -type f -name "config.yaml" -printf '%h\n' | sort -u)

# Re-notify the user if we appended any new missing entries
if [ -s "$OUTPUT_FILE" ]; then
  echo "Missing runs logged to $OUTPUT_FILE"
fi
