#!/bin/bash
#SBATCH --ntasks=24									# Number of tasks (MPI processes)
#SBATCH --cpus-per-task=1							# Number of CPU cores per task
#SBATCH --time=9999:00:00							# Time limit (HH:MM:SS)

cd $SLURM_SUBMIT_DIR

python3 PolyProton_simulation.py