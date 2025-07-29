#!/bin/bash -l
#SBATCH -p xxx
#SBATCH -A xxx
#SBATCH --job-name=1280_64_100.txt
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --output=results/output_1280_64_100.txt
#SBATCH --exclusive


srun -n 64 --mpi=pmi2 --cpu-bind=core python3 run_files/run_parallel_1280_64.py
