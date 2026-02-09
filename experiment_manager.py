from src.driver import GameDriver

from argparse import ArgumentParser
import os
import logging
import yaml
import glob
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
    parser.add_argument("--n_runs", default=5) 
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

        driver = GameDriver(config_path)

        # Set up the replicate folders and run N replicates
        n_runs = args.n_runs
        for run_num in range(n_runs):
            output_dir = os.path.join(root_abs, f"shuffles/shuffle_{run_num}")
            os.makedirs(output_dir, exist_ok=True)

            logging.info("Running config: %s", config_path)
            driver.reset(config_path, exp_folder=output_dir)
            # Check if the result already exists - if so, skip it while ensuring the random number impact is the same (so I guess reset without playing)
            # TODO: A better solution after infilling would be to create a set of random seeds for each replicate.
            has_jsonl_with_10_lines = False

            if os.path.isdir(output_dir):
                for fname in os.listdir(output_dir):
                    if fname.endswith(".jsonl"):
                        path = os.path.join(output_dir, fname)
                        with open(path, "r", encoding="utf-8") as f:
                            if sum(1 for _ in f) == 10:
                                driver.play()
                                break
            
    else:
        raise ValueError("Please specify the --config_path or --config_folder argument. config_folder will run all configs in subdirectories (but not in main directory) so it handles shuffle subdirs")

