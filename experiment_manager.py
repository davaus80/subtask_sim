from src.driver import GameDriver

from argparse import ArgumentParser
import os
import logging
import yaml
import glob


def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--config_path", default=None) 
    parser.add_argument("--config_dir", default=None) 
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
        # Get all configs in subdirectories (skip files in the top-level folder).
        config_paths = []
        root_abs = os.path.abspath(args.config_dir)
        for root, dirs, files in os.walk(root_abs):
            if os.path.abspath(root) == root_abs:
                # skip files in the root folder itself
                continue
            if "config.yaml" in files:
                config_paths.append(os.path.join(root, "config.yaml"))

        if not config_paths:
            logging.warning("No config.yaml files found in subdirectories of %s", args.config_dir)

        # Run all found configs
        for cfg_path in sorted(config_paths):
            logging.info("Running config: %s", cfg_path)
            driver = GameDriver(cfg_path)
            driver.play()
    else:
        raise ValueError("Please specify the --config_path or --config_folder argument. config_folder will run all configs in subdirectories (but not in main directory) so it handles shuffle subdirs")

