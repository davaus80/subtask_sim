#!/bin/bash

# OPTION 1: SUBMIT ALL YAMLS INSIDE A FOLDER, RECURSIVELY
CONFIG_DIR="experiments/basic_novar/name_shuffles"
echo "Searching for configs in $CONFIG_DIR"
# Find all YAML files recursively
CONFIG_FILES=($(find "$CONFIG_DIR" -type f -name "*.yaml" | sort))

# OPTION 2: CUSTOM SELECTION OF CONFIGS
# CONFIGS_FILES=("config1.yaml" "config2.yaml" "config3.yaml")


echo "#### Found the following configs: ####"
for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "$CONFIG"
done

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
for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Submitting job for $CONFIG"
    sbatch run_slurm_job.sh "$CONFIG"
done