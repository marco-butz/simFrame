__author__ = "Marco Butz"

import numpy as np
import sys
import decimal

"""
structure is an ndarray of the following form:
epsilon = np.array([[[1.0 for i in range(dims[2])] for j in range(dims[1])] for k in range(dims[0])])
"""

#correct rounding of halfs:
def myround(num):
    context = decimal.getcontext()
    context.rounding = decimal.ROUND_HALF_UP
    return int(round(decimal.Decimal(num), 0))

#TODO: implement this method efficiently
def setPixel(epsilon: np.ndarray,
                structure: np.ndarray,
                structureScalingFactor: int,
                thickness: int,
                permittivity: float):
                pass

def addCentralPlanarPixelStructure(epsilon: np.ndarray,
                                    structure: np.ndarray,
                                    structureScalingFactor: int,
                                    thickness: int,
                                    permittivity: float):
    """
    does not include highest index:
    [2,5] includes 2,3,4
    """
    dims = epsilon.shape
    zBounds = [int(dims[2]/2 - myround(thickness/2)),int(dims[2]/2 + int(thickness/2))]
    designArea = structure.shape

    if dims[2] == 1:
        zBounds = [0, 1]

    #The following lines are hokus pokus. Don't touch them.
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            l = int((i - (dims[0]/2 - designArea[0]*structureScalingFactor/2)) / structureScalingFactor)
            m = int((j - (dims[1]/2 - designArea[1]*structureScalingFactor/2)) / structureScalingFactor)
            if (i in range(int(dims[0]/2-designArea[0]*structureScalingFactor/2),int(dims[0]/2+designArea[0]*structureScalingFactor/2))
                and j in range(int(dims[1]/2-designArea[1]*structureScalingFactor/2),int(dims[1]/2+designArea[1]*structureScalingFactor/2))):
                if l in range(0,designArea[0]) and m in range(0,designArea[1]):
                    epsilon[i][j][zBounds[0]:zBounds[1]] = [structure[l][m]*(permittivity-1)+1 for o in range(zBounds[1]-zBounds[0])]

    return epsilon


def addPlanar(epsilon: np.ndarray,
                area: [(int, int), (int, int)],
                thickness: int,
                permittivity: float,
                substratePermittivity: float):
    """
    does not include highest index:
    [2,5] includes 2,3,4
    """
    dims = epsilon.shape
    zBounds = [int(dims[2]/2 - myround(thickness/2)),int(dims[2]/2 + int(thickness/2))]

    if dims[2] == 1:
        zBounds = [0,1]

    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            for k in range(0,dims[2]):
                if zBounds[0] <= k < zBounds[1]:
                    if area[0][0] <= i < area[1][0] and area[0][1] <= j < area[1][1]:
                        epsilon[i][j][k] = permittivity

                elif k < zBounds[0]:
                    epsilon[i][j][k] = substratePermittivity

    return epsilon

if __name__ == '__main__':
    print('testrun')
    SiNThickness = 5
    dims = [50, 50, 10]
    designArea = [10, 10]
    structureScalingFactor = 3 #pixels are 120nm vs 40nm sim resulution
    structure = np.ones(100).reshape(designArea[0], designArea[1])
    #structure = np.loadtxt("structures/sometest.txt", delimiter=',')
    epsilon = np.array([[[1.0 for i in range(dims[2])] for j in range(dims[1])] for k in range(dims[0])])
    epsilon = addPlanar(epsilon, [(0, 20),(dims[0]/2, 30)], SiNThickness, 3, 2)
    epsilon = addPlanar(epsilon, [(dims[0]/2, 20),(dims[0], 30)], SiNThickness, 3, 2)
    epsilon = addCentralPlanarPixelStructure(epsilon, structure, structureScalingFactor, SiNThickness, 3)
    #show what's going on:
    for k in range(0,dims[2]):
        print('printing layer ', k, ' ...')
        for j in range(0,dims[1]):
            for i in range(0,dims[0]):
                if epsilon[i][j][k] != 1:
                    sys.stdout.write(str(int(epsilon[i][j][k])))
                else:
                    sys.stdout.write(' ')
            print('')
