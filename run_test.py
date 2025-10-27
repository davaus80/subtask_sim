from src.driver import GameDriver

from argparse import ArgumentParser
import logging
import yaml

def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--config_path", default=None) 
    return parser.parse_args(args)

if __name__ == "__main__":
    # Set up logging
    import logging
    import os

    import sys
    args = get_args(sys.argv[1:])

    config_path = args.config_path

    # Load the YAML config file into a dictionary
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    driver = GameDriver(config)

    driver.play()

