#!/bin/bash

#SBATCH --job-name=gem
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --time=0:10:00
#SBATCH --mem=8G

# Pass config as an argument; optional second arg overrides n_runs (default 10)
CONFIG_DIR=$1
N_RUNS=${2:-10}

module load python/3.10
source ~/skyfall310/bin/activate
export HF_HOME=$SCRATCH/hf_cache

python experiment_manager.py --config_dir $CONFIG_DIR --n_runs $N_RUNS
