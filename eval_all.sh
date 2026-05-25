#!/bin/zsh

toplevel_dirs=(
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260118_farm_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260311_gemini_farm_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260124_farm_2sd_var"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260124_farm_4sd_var"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260120_ab_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260312_gemini_ab_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260124_abandit_2sd_var"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260124_abandit_4sd_var"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260312_rec_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260312_rec_4sd_var"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260312_rec_2sd_var"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260523_ab_warn_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260523_farm_warn_novar"
    "/Users/david/Documents/01_Projects/Skyfall/subtask_sim/experiments/20260523_rec_warn_novar"
)

for toplevel in $toplevel_dirs; do
    for dir in $toplevel/*(/)  ; do
        python ./eval_manager.py --superfolder_path "$dir" --min_rows 10 --output eval_results.csv
    done
done
