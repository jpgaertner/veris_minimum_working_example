#!/bin/bash
#SBATCH --partition=xxx
#SBATCH --account=xxx
#SBATCH --job-name=1024_1x4
#SBATCH --output=out_gpu/out/1024_1_1x4_1000.log
#SBATCH --error=out_gpu/out/1024_1_1x4_1000.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100_80:4
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
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
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(($SLURM_GPUS_ON_NODE * $SLURM_JOB_NUM_NODES))"
echo "Start time: $(date)"
echo "=============================================="
echo ""

PYTHON_PATH=~/.conda/envs/jax/bin/python

# Set environment variables
#export JAX_PLATFORMS=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_ENABLE_X64=True

# try later
#export  XLA_PYTHON_CLIENT_PREALLOCATE=true   # Grab memory upfront
#export  XLA_PYTHON_CLIENT_ALLOCATOR=default  # Keep memory pooled , try cuda_async later

# Debug level (0=minimal, 1=basic, 2=detailed, 3=verbose)
export JAX_DEBUG_LEVEL=1

# JAX Distributed Coordination Setup, use this only for multi-task jobs
# Get the first node as coordinator
if [ "$SLURM_NTASKS" -gt 1 ]; then
    COORDINATOR_HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    COORDINATOR_PORT=12345
    export JAX_COORDINATOR_ADDRESS="${COORDINATOR_HOST}:${COORDINATOR_PORT}"
fi

echo "JAX Distributed Configuration:"
echo "  Coordinator: $JAX_COORDINATOR_ADDRESS"
echo "  Total processes: $SLURM_NTASKS"
echo ""

# activate conda environment
source ~/.bashrc
conda activate jax

SCRIPT=run_files/run_parallel_gpu_1024_1x4.py
echo "Running script: $SCRIPT"
echo ""

srun $PYTHON_PATH $SCRIPT

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
