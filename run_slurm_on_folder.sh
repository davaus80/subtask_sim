#!/bin/bash

#SBATCH --job-name=sky
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --time=0:10:00
#SBATCH --mem=30G

# Pass config as an argument
CONFIG_DIR=$1

module load python/3.10
source ~/skyfall310/bin/activate

export HF_HOME=$SCRATCH/hf_cache

python experiment_manager.py --config_dir $CONFIG_DIR
