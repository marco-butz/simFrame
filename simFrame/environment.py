__author__ = "Marco Butz"

from simFrame.buildStructure import addPlanar, addCentralPlanarPixelStructure
from simFrame.remoteSolver.sendJob import sendSimulationJob
import simFrame.permittivities as permittivities
import numpy as np
import os
import sys

class Environment:

    def __init__(self, dimensions: np.ndarray,
                    designArea: (int, int),
                    structureScalingFactor: int,
                    waveguides: list,
                    thickness: int,
                    wavelength: int,
                    pixelSize: int,
                    structurePermittivity: float,
                    substratePermittivity: float,
                    surroundingPermittivity: float,
                    inputModes: list,
                    outputModes: list):

        self.initialEpsilon = np.array([[[surroundingPermittivity for i in range(dimensions[2])] \
            for j in range(dimensions[1])] \
            for k in range(dimensions[0])])
        self.structure = None
        self.initialStructure = None
        self.structurePermittivity = structurePermittivity
        self.substratePermittivity = substratePermittivity
        self.surroundingPermittivity = surroundingPermittivity
        self.structureScalingFactor = structureScalingFactor
        self.thickness = thickness
        self.pixelSize = pixelSize
        self.dimensions = dimensions
        self.inputModes = inputModes
        self.outputModes = outputModes
        self.wavelength = wavelength

        for i in waveguides:
            self.initialEpsilon = addPlanar(self.initialEpsilon, [i[0], i[1]], thickness, structurePermittivity, substratePermittivity)

        self.epsilon = self.initialEpsilon.copy()

    def setStructure(self, structure: np.ndarray):
        self.epsilon = addCentralPlanarPixelStructure(self.epsilon, structure,
            self.structureScalingFactor, self.thickness, self.structurePermittivity)
        if self.structure == None:
            self.initialStructure = structure.copy()
            self.initialEpsilon = self.epsilon.copy()
        self.structure = structure

    def resetStructure(self):
        self.structure = self.initialStructure.copy()
        self.epsilon = self.initialEpsilon.copy()

    def flipPixel(self, xy: (int, int)):
        self.structure[xy[0], xy[1]] = 0 if self.structure[xy[0], xy[1]] == 1 else 1
        self.epsilon = addCentralPlanarPixelStructure(self.epsilon,
            self.structure, self.structureScalingFactor, self.thickness, self.structurePermittivity)

    def setFOM(self, FOM):
        self.FOM = FOM

    def evaluate(self, plot=0, plotDir=""):
        results = sendSimulationJob(epsilon=self.epsilon,
                                    inputModes=self.inputModes,
                                    outputModes=self.outputModes,
                                    wavelength=self.wavelength,
                                    pixelSize=self.pixelSize,
                                    dims=self.dimensions,
                                    plotDebug=plot,
                                    method="fdtd",
                                    plotDir=plotDir)
        overlaps = []
        for i in range(len(self.inputModes)):
            for j in range(len(self.outputModes)):
                overlaps.append(results[i][j]['overlap'])

        return self.FOM(overlaps)

    def printStructure(self, layerIndex: int = 0):
        for j in range(0,self.dimensions[1]):
            for i in range(0,self.dimensions[0]):
                if self.epsilon[i][j][layerIndex] != 1:
                    sys.stdout.write(str(int(self.epsilon[i][j][0])))
                else:
                    sys.stdout.write(' ')
            print('')

if __name__ == "__main__":

    os.environ["SIMULATE_ON_THIS_MACHINE"] = "1"
    #os.environ["X_USE_MPI"] = "1"
    producePlots = 0

    dims = [150, 100, 1]
    env = Environment(dimensions = dims,
                    designArea = [20, 20],
                    structureScalingFactor = 3,
                    waveguides = [[(0, 44), (dims[0]/2, 56)], [(dims[0]/2, 35), (dims[0], 65)]],
                    thickness = 1,
                    wavelength = 775,
                    pixelSize = 40,
                    structurePermittivity = permittivities.SiN,
                    substratePermittivity = permittivities.SiO,
                    surroundingPermittivity = permittivities.Air,
                    inputModes = [{'pos':[[12,12,0],[12,dims[1]-12,0]], 'modeNum': 0}],
                    outputModes = [{'pos':[[dims[0]-12,12,0],[dims[0]-12,dims[1]-12,0]], 'modeNum': 4}])

    env.setStructure(np.ones(400).reshape(20,20))

    def figureOfMerit(overlaps):
        return overlaps[0][0]

    env.setFOM(figureOfMerit)
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug0" + '/'))

    env.flipPixel([5,8])
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug1" + '/'))

    env.flipPixel([8,8])
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug2" + '/'))

    env.flipPixel([8,8])
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug3" + '/'))

    env.resetStructure()
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug4" + '/'))

    for i in range(100):
        randomGenerator = np.random.default_rng()
        env.flipPixel([randomGenerator.choice(20,1),randomGenerator.choice(20,1)])
    env.printStructure()
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug5" + '/'))

    env.resetStructure()
    for i in range(6,15):
        for j in [9,10]:
            env.flipPixel([i,j])
    env.printStructure()
    print(env.evaluate(plot = producePlots, plotDir = 'simFrame/' + "simulationData/debug6" + '/'))
