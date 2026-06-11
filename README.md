# Veris minimum working example

Scripts to run Veris as a standalone model with benchmark forcing fields. The dynamics and thermodynamics components are separated here but can easily be combined.

This branch uses the redesigned version of Veris ([jax-only branch](https://github.com/jpgaertner/veris/tree/jax-only)), which uses JAX's sharded arrays for parallel execution.

To use these scripts, install Veris from the [Veris Github repository](https://github.com/jpgaertner/veris/tree/jax-only)) (```pip install -e .```, ensure the ```jax-only``` branch is used). Then create the conda environment from ```jax_env.yml```.


### How to use

The initial conditions and forcing fields are defined in ```initialize_growth.py``` and ```initialize_dyn.py```. Veris can be run on a single process in a Jupyter notebook using ```run_dyn.ipynb``` and ```run_growth.ipynb```. For parallel execution, ```run_cpu.sh``` and ```run_gpu.sh``` are used to specify the run settings. These scripts automatically update the job script (```job_cpu.sh``` or ```job_gpu.sh```) and the main routine (```run_parallel.py```), save copies of these files to a ```run_files``` directory (which must exist), and submit the job via SLURM.
