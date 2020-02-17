__author__= "Marco Butz"

import scipy.io as sio
import numpy as np
import random
import subprocess
import string
import os.path
import time
import os
from collections import Iterable
from time import sleep
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

def randomString(stringLength=32):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def sendSimulationJob(epsilon: np.ndarray,
                        inputModes: np.ndarray,
                        outputModes: np.ndarray,
                        wavelength: int,
                        pixelSize: int,
                        dims: np.ndarray,
                        method='fdtd',
                        plotDebug=0,
                        plotDir='simulationData/'):

    omega = 2*np.pi/wavelength
    #if os.environ.get('SIMULATE_ON_THIS_MACHINE') ==  None or method != 'fdtd':
    if os.environ.get('X_USE_MPI') ==  "1" or method != 'fdtd':
        jobs = []
        jobStates = []
        for mode in inputModes:
            jobName = randomString() + '.mat'
            jobs.append(jobName)
            jobStates.append(False) #unfinished
            sio.savemat('./simulationData/' + jobName, {'epsilon': epsilon,
                                    'modeSourcePos': mode['pos'],
                                    'modeSourceNum': mode['modeNum'],
                                    'numModesToMeasure': len(outputModes),
                                    'posModesToMeasure': [i['pos'] for i in outputModes],
                                    'modeNumModesToMeasure': [i['modeNum'] for i in outputModes],
                                    'omega': omega,
                                    'dims': dims,
                                    'pixelSize': pixelSize,
                                    'wavelength': wavelength})

            subprocess.call("remoteSolver/send_job.sh " + jobName + " " + str(plotDebug) + " " + method, shell=True)

        resultOutputModes = [[dict() for _ in range(len(outputModes))] for _ in range(len(inputModes))]
        #print('running ', len(jobStates), ' jobs')
        while not all(jobStates):
            for i,jobName in enumerate(jobs):
                if jobStates[i] != True:
                    if(os.path.isfile('./simulationData/results_' + jobName)):
                        #print('found results_', jobName)
                        time.sleep(0.01)
                        mat = sio.loadmat('./simulationData/results_' + jobName, squeeze_me=True)
                        if not isinstance(mat['modeNum'], Iterable):
                            matModeNum = [mat['modeNum']]
                            matOverlap = [mat['overlap']]
                            matPos = [mat['pos']]
                        else:
                            matModeNum = mat['modeNum']
                            matOverlap = mat['overlap']
                            matPos = mat['pos']
                        for j, modeNum in enumerate(matModeNum):
                            #print('received result for outputmode ', matModeNum[j], ' inputmode: ', mat['inputModeNum'])
                            resultOutputModes[i][j].update({'modeNum': modeNum,
                                                    'overlap': matOverlap[j],
                                                    'inputModeNum': mat['inputModeNum'],
                                                    'inputModePos': mat['inputModePos'],
                                                    'pos': matPos[j]})
                        jobStates[i] = True
                        #os.remove('./simulationData/results_' + jobName)
                        #os.remove('./simulationData/' + jobName)
            sleep(0.05)
    else:
        #we want to directly call the function instead of launching the interpreter again. Unfortunately we need to do this serially
        from .fdtd.simulation import simulation as fdtd
        resultOutputModes = [[dict() for _ in range(len(outputModes))] for _ in range(len(inputModes))]

        #embarassingly parallelize if we dont have to plot. Matplotlib is not threadsafe
        results = []
        if plotDebug == 0:
            mats = []
            pool = ThreadPool(len(inputModes))

            plotMe = plotDebug
            plotDir = plotDir
            jobName = 'direct-'
            for i,mode in enumerate(inputModes):
                mats.append({'epsilon': epsilon,
                        'modeSourcePos': mode['pos'],
                        'modeSourceNum': mode['modeNum'],
                        'numModesToMeasure': len(outputModes),
                        'posModesToMeasure': [i['pos'] for i in outputModes],
                        'modeNumModesToMeasure': [i['modeNum'] for i in outputModes],
                        'omega': omega,
                        'dims': dims,
                        'pixelSize': pixelSize,
                        'wavelength': wavelength})

            func = partial(fdtd, plotMe, plotDir, jobName)
            results = pool.map(func, mats)
            pool.close()
            pool.join()
            for i,mode in enumerate(inputModes):
                for mat in results:
                    if not isinstance(mat['modeNum'], Iterable):
                        matModeNum = [mat['modeNum']]
                        matOverlap = [mat['overlap']]
                        matPos = [mat['pos']]
                    else:
                        matModeNum = mat['modeNum']
                        matOverlap = mat['overlap']
                        matPos = mat['pos']
                    for j, modeNum in enumerate(matModeNum):
                        resultOutputModes[i][j].update({'modeNum': modeNum,
                                                'overlap': matOverlap[j],
                                                'inputModeNum': mat['inputModeNum'],
                                                'inputModePos': mat['inputModePos'],
                                                'pos': matPos[j]})
        else:
            plotMe = plotDebug
            plotDir = plotDir
            jobName = 'direct-'
            for i,mode in enumerate(inputModes):
                results.append(fdtd(plotMe=plotDebug, plotDir=plotDir, jobSpecifier='direct-'+randomString(),
                mat={'epsilon': epsilon,
                    'modeSourcePos': mode['pos'],
                    'modeSourceNum': mode['modeNum'],
                    'numModesToMeasure': len(outputModes),
                    'posModesToMeasure': [i['pos'] for i in outputModes],
                    'modeNumModesToMeasure': [i['modeNum'] for i in outputModes],
                    'omega': omega,
                    'dims': dims,
                    'pixelSize': pixelSize,
                    'wavelength': wavelength}))

                for mat in results:
                    if not isinstance(mat['modeNum'], Iterable):
                        matModeNum = [mat['modeNum']]
                        matOverlap = [mat['overlap']]
                        matPos = [mat['pos']]
                    else:
                        matModeNum = mat['modeNum']
                        matOverlap = mat['overlap']
                        matPos = mat['pos']
                    for j, modeNum in enumerate(matModeNum):
                        resultOutputModes[i][j].update({'modeNum': modeNum,
                                                'overlap': matOverlap[j],
                                                'inputModeNum': mat['inputModeNum'],
                                                'inputModePos': mat['inputModePos'],
                                                'pos': matPos[j]})

    if isinstance(resultOutputModes, dict):
        resultOutputModes = [resultOutputModes]

    return resultOutputModes
