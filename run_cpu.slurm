#!/bin/bash
#SBATCH --ntasks=96									# Number of tasks (MPI processes)
#SBATCH --cpus-per-task=1							# Number of CPU cores per task
#SBATCH --time=9999:00:00							# Time limit (HH:MM:SS)

cd $SLURM_SUBMIT_DIR

#OPENMPI
export "PATH=$PATH:/opt/openmpi/bin"
export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openmpi/lib"

#AMBER
export AMBERHOME=$AMBERHOME:/opt/amber/amber24
export PATH=$PATH:$AMBERHOME/bin
source /opt/amber/amber24/amber.sh

DIR1=`pwd`
DIR2=$RANDOM
UNAME=`whoami`

#copy files into the local /home/Calculations/UNAME directory
mkdir /home/Calculations/$UNAME/$DIR2
cd $DIR1
cp -r * /home/Calculations/$UNAME/$DIR2

#using local Calculations folder
cd /home/Calculations/$UNAME/$DIR2

export USE_GPU=False
python3 PolyProton_simulation.py