__author__= "Marco Butz"

import time

#import matplotlib.pyplot as plt

import meep as mp
import scipy.io as sio
import numpy
import os
import h5py
import sys
import random
from collections.abc import Iterable
from pathlib import Path
import string

def randomString(stringLength=32):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def simulation(plotMe, plotDir='simulationData/', jobSpecifier='direct-', mat=None):
    if os.getenv("X_USE_MPI") != "1":
        jobName = jobSpecifier + randomString()
    else:
        jobName = jobSpecifier
    start = time.time()

    if str(plotMe) == '1':
        os.makedirs(plotDir)
        import matplotlib
        #matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        print('will plot')
    else:
        mp.quiet(True)
    __author__ = 'Marco Butz'

    pixelSize = mat['pixelSize']

    spectralWidth = 300/mat['wavelength']
    modeFrequencyResolution = 1
    normOffset = pixelSize/1000 * 10
    cell = mp.Vector3(mat['dims'][0]*pixelSize/1000, mat['dims'][1]*pixelSize/1000,0)#, mat['dims'][2]*pixelSize/1000)

    #generate hdf5 epsilon file
    if mp.am_master():
        h5f = h5py.File(jobName + '_eps.h5', 'a')
        h5f.create_dataset('epsilon', data=mat['epsilon'])
        h5f.close()

    sourceCenter = [(mat['modeSourcePos'][0][0]+mat['modeSourcePos'][1][0])/2,
                    (mat['modeSourcePos'][0][1]+mat['modeSourcePos'][1][1])/2,
                    (mat['modeSourcePos'][0][2]+mat['modeSourcePos'][1][2])/2]
    sourceSize = [(mat['modeSourcePos'][1][0]-mat['modeSourcePos'][0][0]),
                    (mat['modeSourcePos'][1][1]-mat['modeSourcePos'][0][1]),
                    (mat['modeSourcePos'][1][2]-mat['modeSourcePos'][0][2])]

    modeNumModesToMeasure = []
    posModesToMeasure = []
    if not isinstance(mat['modeNumModesToMeasure'],Iterable):
        #this wraps stuff into an array if it has been squeezed before
        posModesToMeasure = [mat['posModesToMeasure']]
        modeNumModesToMeasure = [mat['modeNumModesToMeasure']]
        print('transformed')
    else:
        posModesToMeasure = mat['posModesToMeasure']
        modeNumModesToMeasure = mat['modeNumModesToMeasure']

    outputsModeNum = []
    outputsCenter = []
    outputsSize = []
    for i in range(0,mat['numModesToMeasure']):
        outputsCenter.append([(posModesToMeasure[i][0][0]+posModesToMeasure[i][1][0])/2,
                        (posModesToMeasure[i][0][1]+posModesToMeasure[i][1][1])/2,
                        (posModesToMeasure[i][0][2]+posModesToMeasure[i][1][2])/2])
        outputsSize.append([(posModesToMeasure[i][1][0]-posModesToMeasure[i][0][0]),
                        (posModesToMeasure[i][1][1]-posModesToMeasure[i][0][1]),
                        (posModesToMeasure[i][1][2]-posModesToMeasure[i][0][2])])
        outputsModeNum.append(modeNumModesToMeasure[i])

    for i in range(0,len(sourceCenter)):
        sourceCenter[i] = sourceCenter[i] * pixelSize / 1000 - cell[i]/2
        sourceSize[i] = sourceSize[i] * pixelSize / 1000
    for i in range(0,len(outputsCenter)):
        for j in range(0,len(outputsCenter[i])):
            outputsCenter[i][j] = outputsCenter[i][j] * pixelSize / 1000 - cell[j]/2
            outputsSize[i][j] = outputsSize[i][j] * pixelSize / 1000

    sources = [mp.EigenModeSource(src=mp.GaussianSource(wavelength=mat['wavelength']/1000,fwidth=spectralWidth),
                                    eig_band=mat['modeSourceNum']+1,
                                    center=mp.Vector3(sourceCenter[0],sourceCenter[1],sourceCenter[2]),
                                    size=mp.Vector3(sourceSize[0],sourceSize[1],sourceSize[2]))]
    """
    sources = [mp.EigenModeSource(src=mp.ContinuousSource(wavelength=mat['wavelength']/1000),
                                    eig_band=mat['modeSourceNum']+1,
                                    center=mp.Vector3(sourceCenter[0],sourceCenter[1],sourceCenter[2]),
                                    size=mp.Vector3(sourceSize[0],sourceSize[1],sourceSize[2]))]
    """

    resolution = 1000/pixelSize #pixels per micrometer

    pmlLayers = [mp.PML(pixelSize*10/1000)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pmlLayers,
                        geometry=[],
                        epsilon_input_file=jobName + '_eps.h5',
                        sources=sources,
                        resolution=resolution)
                        #force_complex_fields=True) #needed for fdfd solver

    transmissionFluxes = []
    transmissionModes = []
    normFluxRegion = mp.FluxRegion(center=mp.Vector3(sourceCenter[0]+normOffset,sourceCenter[1],sourceCenter[2]),
                                size=mp.Vector3(sourceSize[0],sourceSize[1],sourceSize[2]),
                                direction=mp.X)
    normMode = sim.add_mode_monitor(1000/mat['wavelength'], spectralWidth, modeFrequencyResolution, normFluxRegion)
    normFlux = sim.add_flux(1000/mat['wavelength'], spectralWidth, modeFrequencyResolution, normFluxRegion)

    for i in range(0,len(outputsCenter)):
        transmissionFluxRegion = mp.FluxRegion(center=mp.Vector3(outputsCenter[i][0],outputsCenter[i][1],outputsCenter[i][2]),
                                                size=mp.Vector3(outputsSize[i][0],outputsSize[i][1],outputsSize[i][2]),
                                                direction=mp.X)
        transmissionFluxes.append(sim.add_flux(1000/mat['wavelength'], spectralWidth, modeFrequencyResolution, transmissionFluxRegion))
        transmissionModes.append(sim.add_mode_monitor(1000/mat['wavelength'], spectralWidth, modeFrequencyResolution, transmissionFluxRegion))
    if str(plotMe) == '1':
        animation = mp.Animate2D(sim,
                           fields=mp.Ey,
                           realtime=False,
                           normalize=True,
                           field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'},
                           boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})
        sim.run(mp.at_every(0.5,animation),until_after_sources=mp.stop_when_fields_decayed(20,mp.Ey,mp.Vector3(outputsCenter[0][0],outputsCenter[0][1],outputsCenter[0][2]),1e-5))
        #sim.init_sim()
        #sim.solve_cw(tol=10**-5,L=20)
        print('saving animation to ' + str(os.path.join(plotDir + 'animation.gif')))
        animation.to_gif(10, os.path.join(plotDir + 'inputMode_' + str(mat['modeSourceNum']) + '_' + 'animation.gif'))
    else:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(20,mp.Ey,mp.Vector3(outputsCenter[0][0],outputsCenter[0][1],outputsCenter[0][2]),1e-5))

    normModeCoefficients = sim.get_eigenmode_coefficients(normMode, [mat['modeSourceNum']+1], direction=mp.X)
    #print('input norm coefficients TE00: ', numpy.abs(sim.get_eigenmode_coefficients(normMode, [1], direction=mp.X).alpha[0][0][0])**2)
    #print('input norm coefficients TE10: ', numpy.abs(sim.get_eigenmode_coefficients(normMode, [3], direction=mp.X).alpha[0][0][0])**2)
    #print('input norm coefficients TE20: ', numpy.abs(sim.get_eigenmode_coefficients(normMode, [5], direction=mp.X).alpha[0][0][0])**2)
    #normFluxes = sim.get
    resultingModes = []
    resultingOverlaps = []
    for i in range(0,len(outputsCenter)):
        resultingModes.append(sim.get_eigenmode_coefficients(transmissionModes[i], [outputsModeNum[i]+1], direction=mp.X))
        resultingOverlaps.append([numpy.abs(resultingModes[i].alpha[0][j][0])**2/numpy.abs(normModeCoefficients.alpha[0][j][0])**2 for j in range(modeFrequencyResolution)])
        #resultingFluxes.append(sim.get_flux_data(transmissionFluxes[i]) / inputFlux)

    if str(plotMe) == '1':
        eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

        plt.figure()
        for i in range(0,len(resultingModes)):
            frequencys = numpy.linspace(1000/mat['wavelength'] - spectralWidth/2, 1000/mat['wavelength'] + spectralWidth/2, modeFrequencyResolution)
            plt.plot(1000/frequencys, resultingOverlaps[i], label='Transmission TE' + str(int(outputsModeNum[i]/2)) + '0')
            print('mode coefficients: ' + str(resultingOverlaps[i]) + ' for mode number ' + str(outputsModeNum[i]))
            print('mode coefficients: ' + str(resultingModes[i].alpha[0]) + ' for mode number ' + str(outputsModeNum[i]))
        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.savefig(os.path.join(plotDir + 'inputMode_' + str(mat['modeSourceNum']) + '_' + 'mode_coefficients.png'))
        plt.close()

        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig(os.path.join(plotDir + 'inputMode_' + str(mat['modeSourceNum']) + '_' + 'debug_structure.png'))
        plt.close()

        inputFourier = [sources[0].src.fourier_transform(1000/f) for f in range(1,1000)]
        plt.figure()
        plt.plot(inputFourier)
        plt.savefig(os.path.join(plotDir + 'inputMode_' + str(mat['modeSourceNum']) + '_' + 'debug_input_fourier.png'))
        plt.close()

        ez_data = numpy.real(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez))
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig(os.path.join(plotDir + 'inputMode_' + str(mat['modeSourceNum']) + '_' + 'debug_overlay.png'))
        plt.close()

    #it might be possible to just reset the structure. will result in speedup
    mp.all_wait()
    sim.reset_meep()
    end = time.time()
    if mp.am_master():
        os.remove(jobName + '_eps.h5')
        print('simulation took ' + str(end - start))

    if __name__ == "__main__":
        jobNameWithoutPath = jobName.split('/')[len(jobName.split('/'))-1]
        sio.savemat("results_" + jobNameWithoutPath, {'pos': posModesToMeasure,
                                        'modeNum': modeNumModesToMeasure,
                                        'overlap': resultingOverlaps,
                                        'inputModeNum': mat['modeSourceNum'],
                                        'inputModePos': mat['modeSourcePos']})
    else:
        return {'pos': posModesToMeasure,
                'modeNum': modeNumModesToMeasure,
                'overlap': resultingOverlaps,
                'inputModeNum': mat['modeSourceNum'],
                'inputModePos': mat['modeSourcePos']}

if __name__ == "__main__":
    #usage: simulation.py jobName.mat plotMe
    jobName = sys.argv[1]
    plotMe = sys.argv[2]
    mat = sio.loadmat(jobName, squeeze_me=True)
    simulation(plotMe=0, plotDir='simulationData/', jobSpecifier=jobName, mat=mat)
