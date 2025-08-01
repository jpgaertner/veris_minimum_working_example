
proc_grids=("(1, 1)" "(2, 2)" "(2, 4)" "(4, 4)" "(4, 8)" "(8, 8)")
grid_lens=(1024)
n_timesteps=50

# set processor type (compute for cpu or gpu for gpu)
partition="compute"

# apply timesteps and processor type
sed -i "s|\(n_timesteps = \).*|\1$n_timesteps|" run_parallel.py
sed -i "s|\(#SBATCH -p \).*|\1$partition|" job.sh

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
        cp run_parallel.py run_files/run_parallel_${grid_len}_${nproc}.py

        # set number of processors
        sed -i "s|\(#SBATCH --nodes=\).*|\1$nproc|" job.sh
        #sed -i "s|\(#SBATCH --nodes=\).*|\11|" job.sh
        sed -i "s|\(#SBATCH --ntasks=\).*|\1$nproc|" job.sh

        # partitioning dependent syntax
        if [ "$partition" = "gpu" ]; then
          sed -i "8s|.*|#SBATCH --gpus-per-task=1|" job.sh
        else
          sed -i "8s|.*||" job.sh
        fi

        # update names of output, the used run_parallel, and job
        sed -i "s|\(#SBATCH --output=results/output_\).*|\1${grid_len}_${nproc}_${n_timesteps}.txt|" job.sh
        sed -i "s|\(srun --mpi=pmi2 \).*|\1python3 run_files/run_parallel_${grid_len}_${nproc}.py|" job.sh
        sed -i "s|\(#SBATCH --job-name=\).*|\1${grid_len}_${nproc}_${n_timesteps}|" job.sh


        cp job.sh run_files/job_${grid_len}_${nproc}_${n_timesteps}.sh

        sbatch run_files/job_${grid_len}_${nproc}_${n_timesteps}.sh
    done
done
