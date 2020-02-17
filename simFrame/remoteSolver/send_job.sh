#!/bin/bash

#$1 has to be local temp file name
#$2 1 for ploting
#$3 fdfd or fdtd

if [ "$1" == "" ]; then
  exit 1
fi
#path of script
SPATH=$(pwd)

if [ ! -z "$SIMULATE_ON_THIS_MACHINE" ]; then
  bash remoteSolver/localSimulation.sh $1 $2 $3 &

else
  #execute slurm config stuff
  sbatch --job-name=Neuro --output=simulationData/slurm_%j.out --ntasks 32 --mem 64000 --partition UltraShort --time=0:29:00 $SPATH/remoteSolver/slurmConfig.pbs $SPATH/simulationData/$1 $1 $2 $3
fi
