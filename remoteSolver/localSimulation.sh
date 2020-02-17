#!/bin/bash

#$1 has to be local temp file name
#$2 1 for ploting
#$3 fdfd or fdtd

if [ "$3" == "fdfd" ]; then
  python3 remoteSolver/fdfd/simulation.py simulationData/$1 $2 > logs/simulationLog_$1.txt
  rm simulationData/$1
  mv results_$1 simulationData/results_$1
fi
if [ "$3" == "fdtd" ]; then
  mpirun -np 8 python3 remoteSolver/fdtd/simulation.py simulationData/$1 $2 > logs/simulationLog_$1.txt
  rm simulationData/$1
  mv results_$1 simulationData/results_$1
fi
