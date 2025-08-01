#!/bin/bash -l
#SBATCH -p xxx
#SBATCH -A xxx
#SBATCH --job-name=1280_64_100
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64

#SBATCH --output=results/output_1280_64_100.txt
#SBATCH --exclusive

export JAX_ENABLE_X64=True

srun --mpi=pmi2 --cpu-bind=core python3 run_files/run_parallel_1024_128.py
