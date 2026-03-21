from src.driver import GameDriver

from argparse import ArgumentParser
import os
import logging
import yaml
import glob
import json
import random


'''
Experiment_manager.py is where we manage the execution of experiments. 
We expect the following structure:
----------------------------------------------------------------------
top_level_experiment folder/
|- superfolder_0/
|  |- semantic_variation_0/
|  |  |- config.yaml
|  |  |- shuffles/
|  |  |  |- shuffle_0
|  |  |  |  |- run_YYYYMMDD_HHMMSS.log
|  |  |  |  |- run_YYYYMMDD_HHMMSS.jsonl
|  |  |  |- ...
|  |  |  |- shuffle_(N-1)
|  |- semantic_variation_1/
|  |- ...
|  |- semantic_variation_(K-1)/
|- superfolder_1/
|- ...
|- superfolder_(M-1)/
----------------------------------------------------------------------

There are two ways to run the experiment manager.
1) --config_dir top_level_experiment/superfolder_i/semantic__variation_j
This option will run N replicates of a semantic variation folder. 

2) --config_path top_level_experiment/superfolder_i/semantic_variation_j/config.yaml
This option will run a single execution semantic variation folder. It will run it for N replicates (specified in the config)

In both cases, we create the shuffles folder and subfolders if they doesn't exist.
'''

def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--config_path", default=None) 
    parser.add_argument("--config_dir", default=None) 
    parser.add_argument("--n_runs", default=10) 
    return parser.parse_args(args)

if __name__ == "__main__":
    # Set up logging
    import logging
    import os

    import sys
    args = get_args(sys.argv[1:])
    
    if args.config_path:
        driver = GameDriver(args.config_path)
        driver.play()
    elif args.config_dir:
        # Set random seed
        seed = 42
        random.seed(seed)
        
        # Get the config
        root_abs = os.path.abspath(args.config_dir)
        config_path = os.path.join(root_abs, "config.yaml")
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        driver = GameDriver(config_path)

        n_runs = config.get('experiment', {}).get('replicates', 10)

        # Generate seeds for each run
        seeds = [random.randint(0, 2**32 - 1) for _ in range(n_runs)]

        # Set up the replicate folders and run N replicates
        for run_num in range(n_runs):
            # Reset the random seed for this run
            random.seed(seeds[run_num])

            output_dir = os.path.join(root_abs, f"shuffles/shuffle_{run_num}")
            os.makedirs(output_dir, exist_ok=True)

            logging.info("Running config: %s", config_path)
            driver.reset(config_path, exp_folder=output_dir)

            time_horizon = config.get('experiment', {}).get('time_horizon', 10)
            is_scalesweep = config.get('experiment', {}).get('scalesweep', False)
            has_sufficient_result = False

            for fname in os.listdir(output_dir):
                if not fname.endswith(".jsonl"):
                    continue
                path = os.path.join(output_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) == 0:
                    continue  # failed run
                if is_scalesweep:
                    # A scale sweep run has sufficient results if it has more than one unique action_names,
                    # Since those runs
                    action_names = {json.loads(line).get("action_selected_name") for line in lines}
                    if len(lines) >= time_horizon or len(action_names) > 1:
                        has_sufficient_result = True
                        break
                else:
                    if len(lines) >= time_horizon:
                        has_sufficient_result = True
                        break

            if not has_sufficient_result:
                if is_scalesweep:
                    print("Scalesweep is enabled. Executing scalesweep logic...")
                    driver.play_scalesweep()
                else:
                    driver.play()
            
    else:
        raise ValueError("Please specify the --config_path or --config_folder argument. config_folder will run all configs in subdirectories (but not in main directory) so it handles shuffle subdirs")

