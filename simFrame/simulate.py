__author__= "Marco Butz"

from buildStructure import *
from remoteSolver.sendJob import *
import sys
import os
import numpy as np
import matplotlib.pyplot as pyplot
from datetime import datetime


#usage: python3 simulate.py structureDelimitedByComma methodEither_fdfd_or_fdtd

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

SiNThickness = 1
#dims = [300, 221, 1]
structure = np.loadtxt(sys.argv[1], delimiter=',')
plotDebug = 1

structureScalingFactor = 3

#designArea = [71, 121, 1]
dims = [240, 110, 1]
designArea = [20, 20]

inputModes = [{'pos':[[12,12,0],[12,dims[1]-12,0]], 'modeNum': 0}]

outputModes = [{'pos':[[dims[0]-12,12,0],[dims[0]-12,dims[1]-12,0]], 'modeNum': 4}]
"""
inputModes = [{'pos':[[22,22,0],[22,dims[1]-22,0]], 'modeNum': 1}]

outputModes = [{'pos':[[dims[0]-50,22,0],[dims[0]-50,dims[1]-22,0]], 'modeNum': 1},
                {'pos':[[dims[0]-50,22,0],[dims[0]-50,dims[1]-22,0]], 'modeNum': 3},
                {'pos':[[dims[0]-50,22,0],[dims[0]-50,dims[1]-22,0]], 'modeNum': 5}]
"""
"""
inputModes = [{'pos':[[12,12,0],[12,dims[1]-12,0]], 'modeNum': 0},
                {'pos':[[12,12,0],[12,dims[1]-12,0]], 'modeNum': 4}]

outputModes = [{'pos':[[dims[0]-12,dims[1]/2-15-12,0],[dims[0]-12,dims[1]/2-15+12,0]], 'modeNum': 0},
                {'pos':[[dims[0]-12,dims[1]/2+15-12,0],[dims[0]-12,dims[1]/2+15+12,0]], 'modeNum': 0}]
"""


epsilon = buildStructure(SiNThickness=SiNThickness,dims=dims,
    designArea=designArea,structureScalingFactor=structureScalingFactor,structure=structure,testStraightWG=False)

dataDir = 'simulationData/' + sys.argv[1].split('.')[0].split('/')[-1] + "_" + datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")
if plotDebug == 1:
    os.makedirs(dataDir)
    print('will plot now')
    def pcolor(v):
        vmax = np.max(np.abs(v))
        pyplot.pcolor(v, cmap='seismic', vmin=-vmax, vmax=vmax)
        pyplot.axis('equal')
        pyplot.colorbar()
    pyplot.figure()
    pcolor(np.real(epsilon[:, :, 0]))
    pyplot.savefig(dataDir + '/eps_generated.png')

#the following method is blocking, which means it does not return until every simulation is finished
results = sendSimulationJob(epsilon=epsilon,
                            inputModes=inputModes,
                            outputModes=outputModes,
                            wavelength=775,
                            pixelSize=40,
                            dims=dims,
                            plotDebug=plotDebug,
                            method=sys.argv[2],
                            plotDir='simFrame/' + dataDir + '/') #need to put the extra 'simFrame/' because simulation plotting will go up two dirs

print('outputModes: ', len(outputModes))
effi = []
for i in range(len(inputModes)):
    for j in range(len(outputModes)):
        print('overlap: ', results[i][j]['overlap'], ' for inputMode ', i)
        effi.append(results[i][j]['overlap'])

for i in effi:
    print(i)
