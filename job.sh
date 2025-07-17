#!/bin/bash -l
#SBATCH -p xxx
#SBATCH -A xxx
#SBATCH --job-name=test
#SBATCH -t 0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --output=output.txt


srun -n 4 --mpi=pmi2 --cpu-bind=core python3 parallel_hello_world.py
