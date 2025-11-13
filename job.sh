#!/bin/bash -l
#SBATCH -p xxx
#SBATCH -A xxx
#SBATCH --job-name=256_2_10
#SBATCH -t 0:05:00
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=results/results/output_256_002_10.txt
##SBATCH --exclusive
#SBATCH --mem=480G


export JAX_ENABLE_X64=True

source ~/.bashrc
conda activate veris
# this is necessary for using multiple gpus
module load openmpi/4.1.5-nvhpc-24.9

mpirun python3 -u run_files/run_parallel_256_002.py
