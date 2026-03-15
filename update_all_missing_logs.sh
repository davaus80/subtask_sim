#!/bin/bash

# List of folder names to process
folders=(
    "experiments/20260120_ab_novar"
    "experiments/20260124_abandit_2sd_var"
    "experiments/20260124_abandit_4sd_var"
    "experiments/20260118_farm_novar"
    "experiments/20260124_farm_2sd_var"
    "experiments/20260124_farm_4sd_var"
    "experiments/20260312_rec_novar"
    "experiments/20260312_rec_4sd_var"
    "experiments/20260312_rec_2sd_var"
    # "experiments/20260225_ab_scalesweep_novar"
    # "experiments/20260225_farm_scalesweep_novar"
)

# Loop through each folder and execute the script
for folder in "${folders[@]}"; do
    echo "Processing $folder"
    ./find_incomplete_runs.sh "$folder"
done