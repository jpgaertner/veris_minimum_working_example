#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A bk1377
#SBATCH --job-name=test
#SBATCH -t 0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
##SBATCH --gpus-per-task=1
#SBATCH --output=test.txt


srun --mpi=pmi2 python3 parallel_hello_world.py
