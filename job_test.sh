#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A bk1377
#SBATCH --job-name=test
#SBATCH -t 0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --output=test.txt


mpirun python3 parallel_hello_world.py
