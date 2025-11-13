
#proc_grids=("(1, 1)" "(1, 2)" "(2, 2)" "(2, 4)" "(4, 4)" "(4, 8)" "(8, 8)" "(8, 16)" "(16, 16)" "(16, 32)")
proc_grids=("(1, 2)")
grid_lens=(1024)
n_timesteps=50
mem="480G"
time="0:01:00"

# set processor type (compute for cpu or gpu for gpu)
partition="gpu"

# apply timesteps, processor type and allocated memory
sed -i "s|\(n_timesteps = \).*|\1$n_timesteps|" run_parallel.py
sed -i "s|\(#SBATCH -p \).*|\1$partition|" job.sh
sed -i "s|\(#SBATCH --mem=\).*|\1$mem|" job.sh
sed -i "s|\(#SBATCH -t \).*|\1$time|" job.sh

# used for formatting the run- and output files
format_number() {
    printf "%03d" $1
}

for grid_len in "${grid_lens[@]}"; do

    # set size of benchmark grid
    sed -i "s|\(grid_len = \).*|\1$grid_len|" initialize_dyn.py

    cp initialize_dyn.py run_files/initialize_dyn_${grid_len}.py
    sed -i "s|\(from initialize_dyn\).*|\1_$grid_len import vs, sett|" run_parallel.py
    
    for proc_grid in "${proc_grids[@]}"; do

        # set partitioning
        sed -i "s|\(pdims = \).*|\1$proc_grid|" run_parallel.py

        # read the number of processor from the processor grid
        IFS=',' read -r p0 p1 <<< "${proc_grid//[()]/}"
        nproc=$((p0 * p1))
        nproc_format=$(format_number $nproc)
        cp run_parallel.py run_files/run_parallel_${grid_len}_${nproc_format}.py

        # define maximum number of processor on one node
        if [ "$partition" = "gpu" ]; then
            #max_proc_per_node=4
            max_proc_per_node=1
        else
            #max_proc_per_node=128
            #use max_proc_per_node smaller 32 for the 3072 grid
            max_proc_per_node=8
            #max_proc_per_node=1
        fi
        
        # set number of nodes and processors
        nodes=$(( nproc > max_proc_per_node ? nproc / max_proc_per_node : 1 ))
        proc_per_node=$(( nproc > max_proc_per_node ? max_proc_per_node : nproc ))
        sed -i "s|\(#SBATCH --nodes=\).*|\1$nodes|" job.sh
        sed -i "s|\(#SBATCH --tasks-per-node=\).*|\1$proc_per_node|" job.sh

        # partitioning dependent syntax
        if [ "$partition" = "gpu" ]; then
          sed -i "8s|.*|#SBATCH --gpus-per-task=1|" job.sh
          #run="mpirun"
          run="mpirun --distribution=plane=2"
        else
          sed -i "8s|.*||" job.sh
          run="srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic:cyclic --mem-bind=local"
        fi

        # update names of output, the used run_parallel, and job 
        sed -i "s|\(#SBATCH --output=results/results/output_\).*|\1${grid_len}_${nproc_format}_${n_timesteps}.txt|" job.sh
        sed -i "s|.*\(python3\).*|$run python3 -u run_files/run_parallel_${grid_len}_${nproc_format}.py|" job.sh
        sed -i "s|\(#SBATCH --job-name=\).*|\1${grid_len}_${nproc}_${n_timesteps}|" job.sh

        cp job.sh run_files/job_${grid_len}_${nproc_format}_${n_timesteps}.sh

        sbatch run_files/job_${grid_len}_${nproc_format}_${n_timesteps}.sh
    done
done
