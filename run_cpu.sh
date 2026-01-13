#!/usr/bin/env bash

# settings
proc_grids=("(16, 1)")
max_tasks_per_node=128
cpus_per_task=1
grid_lens=(128 256)
n_timesteps=1000
mem="480GB"
time="0:30:00"


#use max_tasks_per_node=16 or smaller for the 3072 grid
####################################################################

# apply timesteps, processor type and allocated memory
sed -i "s|\(n_timesteps = \).*|\1$n_timesteps|" run_parallel.py
sed -i "s|\(#SBATCH --mem=\).*|\1$mem|" job_cpu.sh
sed -i "s|\(#SBATCH -t \).*|\1$time|" job_cpu.sh

for grid_len in "${grid_lens[@]}"; do

    # set size of benchmark grid
    sed -i "s|\(grid_len = \).*|\1$grid_len|" initialize_dyn.py

    cp initialize_dyn.py run_files/initialize_dyn_${grid_len}.py
    sed -i "s|\(from initialize_dyn\).*|\1_$grid_len import vs, sett|" run_parallel.py
    
    for proc_grid in "${proc_grids[@]}"; do

        # set shape of processor grid and output path
        sed -i "s|\(output_path = \).*|\1'out_cpu/out/'|" run_parallel.py
        sed -i "s|\(pdims = \).*|\1$proc_grid|" run_parallel.py

        # read the number of processor from the processor grid
        IFS=', ' read -r nproc_y nproc_x <<< "${proc_grid//[()]/}"
        nproc=$((nproc_y * nproc_x))

        # copy run file and set name accordingly in job script
        cp run_parallel.py run_files/run_parallel_cpu_${grid_len}_${nproc_y}x${nproc_x}.py
        sed -i "s|.*\(SCRIPT=\).*|\1run_files/run_parallel_cpu_${grid_len}_${nproc_y}x${nproc_x}.py|" job_cpu.sh

        # set number of nodes and processors
        # (use the commented lines only for single task runs)
        nodes=$(( nproc > max_tasks_per_node ? nproc / max_tasks_per_node : 1 ))
        tasks_per_node=$(( nproc > max_tasks_per_node ? max_tasks_per_node : nproc ))
        sed -i "s|\(#SBATCH --nodes=\).*|\1$nodes|" job_cpu.sh
        sed -i "s|\(#SBATCH --ntasks-per-node=\).*|\1$tasks_per_node|" job_cpu.sh
        #sed -i "s|\(#SBATCH --ntasks-per-node=\).*|\11|" job_cpu.sh
        sed -i "s|\(#SBATCH --cpus-per-task=\).*|\1$cpus_per_task|" job_cpu.sh
        #sed -i "s|\(#SBATCH --cpus-per-task=\).*|\1$tasks_per_node|" job_cpu.sh

        # set name of job and output files
        output_file="${grid_len}_${nodes}_${nproc_y}x${nproc_x}_$n_timesteps"
        sed -i "s|\(#SBATCH --job-name=\).*|\1${grid_len}_${nproc_y}x${nproc_x}|" job_cpu.sh
        sed -i "s|\(#SBATCH --output=out_cpu/out/\).*|\1$output_file.log|" job_cpu.sh
        sed -i "s|\(#SBATCH --error=out_cpu/out/\).*|\1$output_file.err|" job_cpu.sh

        # copy job script and run it
        cp job_cpu.sh run_files/job_cpu_${grid_len}_${nproc_y}x${nproc_x}.sh
        sbatch run_files/job_cpu_${grid_len}_${nproc_y}x${nproc_x}.sh
    done
done
