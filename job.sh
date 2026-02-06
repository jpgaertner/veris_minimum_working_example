#!/bin/bash
#SBATCH -p xxx
#SBATCH -A xxx
#SBATCH --job-name=cpu
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=output.txt
#SBATCH --mem=480GB
#SBATCH --exclusive

PYTHON_PATH=~/.conda/envs/jax/bin/python

# activate conda environment
source ~/.bashrc
conda activate jax

srun --cpu-bind=cores --distribution=block:cyclic:cyclic --mem-bind=local $PYTHON_PATH run_dyn.py
