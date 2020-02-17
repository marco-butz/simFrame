simFrame
======

simFrame is a pixel-structure simulation environment.

## Installation

Include pip dependecy in your conda .yml file and add this repo as a pip dependency:
```
- pip:
     - "git+https://zivgitlab.uni-muenster.de/ag-pernice/simframe.git@master#egg=simFrame"
```
Additionally include the following dependencies in your conda .yml:
```
channels:
   - conda-forge
   - clawpack
   - nodefaults
dependencies:
   - python=3.7
   - pymeep=*=mpi_mpich_*
   - hdf5
   - scipy
```

## Sample usage:

```python
from simFrame.environment import Environment
import simFrame.permittivities as permittivities
import numpy as np
import os

#set local simulation:
os.environ["SIMULATE_ON_THIS_MACHINE"] = "1"

#activate MPI:
#no support for parallel hdf5 in v0.1.0
#os.environ["X_USE_MPI"] = "1"

producePlots = 0

#set size of simulation grid [x, y, z]
dims = [150, 100, 1]

env = Environment(dimensions = dims,
                designArea = [20, 20], #the area that is to be changed pixel wise
                structureScalingFactor = 3, #pixels in designArea will be scaled up by this factor for the simulation.
                waveguides = [[(0, 44), (dims[0]/2, 56)], [(dims[0]/2, 35), (dims[0], 65)]], #Array with components [(x1,y1), (x2,y2)]
                thickness = 1, #thickness of designArea. '1' for 2D
                wavelength = 775, #lambda in nm
                pixelSize = 40, #pixelSize for simulation in nm
                structurePermittivity = permittivities.SiN, #permittivity of the structure
                substratePermittivity = permittivities.SiO, #permittivity of the substrate
                surroundingPermittivity = permittivities.Air, #permittivity of the surrounding (typically air)
                inputModes = [{'pos':[[12,12,0],[12,dims[1]-12,0]], 'modeNum': 0}], #array of input Modes. modeNum 0 is TE00
                outputModes = [{'pos':[[dims[0]-12,12,0],[dims[0]-12,dims[1]-12,0]], 'modeNum': 4}]) #array of output Modes. modeNum 4 is TE20 in 2D

#set initial structure:
#takes nparray with shape designArea (here [20,20])
env.setStructure(np.ones(400).reshape(20,20))

#define a function that calculates the figure of merit:
#function has to expect one parameter 'overlaps'. 'overlaps' is a nested list with len(overlaps) = len(inputModes).
#the environment calculates the overlap of each inputmode with all defined outputmodes. i.e. len(overlaps[i]) = len(outputModes).
#the order of the modes passed is preserved.
def figureOfMerit(overlaps):
    return overlaps[0][0]

#set the figure of merit:
env.setFOM(figureOfMerit)

#env.evalute(...) performs the simulations and returns the figure of merit.
#the environment performs len(inputModes) simulations.
#plot = 0 does not produce plots for the simulations.
#plot = 1 does. Warning: HUGE overhead. Just use this for debugging.
print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug0" + '/'))

#flip pixel at coordinates [x,y]:
env.flipPixel([5,8])
print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug1" + '/'))

env.flipPixel([8,8])
print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug2" + '/'))

env.flipPixel([8,8])
print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug3" + '/'))

#reset the structure to the fist structure submittet via setStructure
env.resetStructure()
print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug4" + '/'))
```