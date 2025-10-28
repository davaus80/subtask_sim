#!/bin/bash

#SBATCH --job-name=sky
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=23Gb

# Pass config as an argument
CONFIG_FILE=$1

module load python/3.10
source ~/skyfall310/bin/activate

python experiment_manager.py --config_path $CONFIG_FILE