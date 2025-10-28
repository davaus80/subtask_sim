from src.driver import GameDriver

from argparse import ArgumentParser
import logging
import yaml
import glob


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
    
    if not args.config_path:
        raise ValueError("Please specify the --config_path argument.")
    
    driver = GameDriver(args.config_path)
    driver.play()
