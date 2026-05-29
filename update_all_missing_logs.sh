#!/bin/bash

# List of folder names to process
folders=(
    # "experiments/20260120_ab_novar"
    # "experiments/20260124_abandit_2sd_var"
    # "experiments/20260124_abandit_4sd_var"
    # "experiments/20260118_farm_novar"
    # "experiments/20260124_farm_2sd_var"
    # "experiments/20260124_farm_4sd_var"
    # "experiments/20260312_rec_novar"
    # "experiments/20260312_rec_4sd_var"
    # "experiments/20260312_rec_2sd_var"
    # "experiments/20260311_gemini_farm_novar"
    # "experiments/20260312_gemini_ab_novar"
    # "experiments/20260320_ab_scalesweep_novar"
    # "experiments/20260320_farm_scalesweep_novar"
    # "experiments/20260320_clothing_scalesweep_novar"
    # "experiments/20260523_farm_warn_novar"
    # "experiments/20260523_ab_warn_novar"
    # "experiments/20260523_rec_warn_novar"
    # "experiments/20260524_farm_warn_noexp_novar"
    # "experiments/20260524_ab_warn_noexp_novar"
    "experiments/20260524_rec_warn_noexp_novar"
    # "experiments/20260526_ab_5arm"
    # "experiments/20260526_farm_5arm"
    # "experiments/20260526_rec_5arm"
)

# Loop through each folder and execute the script
for folder in "${folders[@]}"; do
    echo "Processing $folder"
    ./find_incomplete_runs.sh "$folder"
    # ./find_incomplete_scalesweep_runs.sh "$folder"
done