#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A bk1377
#SBATCH --job-name=128_16x1
#SBATCH -t 0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=out_cpu/out/128_1_16x1_1000.log
#SBATCH --error=out_cpu/out/128_1_16x1_1000.err
#SBATCH --mem=480GB
#SBATCH --exclusive

# Print job information
echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo ""

PYTHON_PATH=~/.conda/envs/jax/bin/python

# Set environment variables
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_ENABLE_X64=True

# Debug level (0=minimal, 1=basic, 2=detailed, 3=verbose)
export JAX_DEBUG_LEVEL=1

# JAX Distributed Coordination Setup, use this only for multi-node jobs
# Get the first node as coordinator
if [ "$SLURM_NTASKS" -gt 1 ]; then
    COORDINATOR_HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    COORDINATOR_PORT=12345
    export JAX_COORDINATOR_ADDRESS="${COORDINATOR_HOST}:${COORDINATOR_PORT}"
fi

# activate conda environment
source ~/.bashrc
conda activate jax

SCRIPT=run_files/run_parallel_cpu_128_16x1.py

srun --cpu-bind=cores --distribution=block:cyclic:cyclic --mem-bind=local $PYTHON_PATH $SCRIPT
