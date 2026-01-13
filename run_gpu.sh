#!/usr/bin/env bash

# settings
gpu_grids=("(1, 4)")
max_tasks_per_node=4
grid_lens=(128 256 512 1024)
n_timesteps=1000
mem="480GB"
time="0:10:00"


####################################################################

sed -i "s|\(n_timesteps = \).*|\1$n_timesteps|" run_parallel.py
sed -i "s|\(#SBATCH --mem=\).*|\1$mem|" job_gpu.sh
sed -i "s|\(#SBATCH --time=\).*|\1$time|" job_gpu.sh

for grid_len in "${grid_lens[@]}"; do

    # set size of benchmark grid
    sed -i "s|\(grid_len = \).*|\1$grid_len|" initialize_dyn.py
    cp initialize_dyn.py run_files/initialize_dyn_${grid_len}.py
    sed -i "s|\(from initialize_dyn\).*|\1_$grid_len import vs, sett|" run_parallel.py

    for gpu_grid in "${gpu_grids[@]}"; do
    
        # set shape of gpu grid and output path
        sed -i "s|\(pdims = \).*|\1$gpu_grid|" run_parallel.py
        sed -i "s|\(output_path = \).*|\1'out_gpu/out/'|" run_parallel.py

        # read number of gpus
        IFS=', ' read -r n_gpus_y n_gpus_x <<< "${gpu_grid//[()]/}"
        n_gpus=$((n_gpus_y * n_gpus_x))

        # copy run file and set name accordingly in job script
        cp run_parallel.py run_files/run_parallel_gpu_${grid_len}_${n_gpus_y}x${n_gpus_x}.py
        sed -i "s|.*\(SCRIPT=\).*|\1run_files/run_parallel_gpu_${grid_len}_${n_gpus_y}x${n_gpus_x}.py|" job_gpu.sh

        # set number of nodes and gpus per node
        nodes=$(( n_gpus > max_tasks_per_node ? n_gpus / max_tasks_per_node : 1 ))
        tasks_per_node=$(( n_gpus > max_tasks_per_node ? max_tasks_per_node : n_gpus ))
        cpus_per_task=$(( tasks_per_node * 4 )) # use this only for single-task runs (see below)
        sed -i "s|\(#SBATCH --nodes=\).*|\1$nodes|" job_gpu.sh
        sed -i "s|\(#SBATCH --ntasks-per-node=\).*|\1$tasks_per_node|" job_gpu.sh
        sed -i "s|\(#SBATCH --cpus-per-task=\).*|\14|" job_gpu.sh
        #sed -i "s|\(#SBATCH --cpus-per-task=\).*|\1$cpus_per_task|" job_gpu.sh
        sed -i "s|\(#SBATCH --gres=gpu:a100_80:\).*|\1$tasks_per_node|" job_gpu.sh

        # set name of job and output files
        output_file="${grid_len}_${nodes}_${n_gpus_y}x${n_gpus_x}_$n_timesteps"
        sed -i "s|\(#SBATCH --job-name=\).*|\1${grid_len}_${n_gpus_y}x${n_gpus_x}|" job_gpu.sh
        sed -i "s|\(#SBATCH --output=out_gpu/out/\).*|\1$output_file.log|" job_gpu.sh
        sed -i "s|\(#SBATCH --error=out_gpu/out/\).*|\1$output_file.err|" job_gpu.sh

        # copy job script and run it
        cp job_gpu.sh run_files/job_gpu_${grid_len}_${n_gpus_y}x${n_gpus_x}.sh
        sbatch run_files/job_gpu_${grid_len}_${n_gpus_y}x${n_gpus_x}.sh
    done
done
