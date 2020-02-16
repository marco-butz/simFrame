import subprocess
import sys
from importlib import util
import time

import scipy.sparse.linalg


def module_available(name):
    return util.find_spec(name) is not None

if module_available('opencl_fdfd'):
    import opencl_fdfd as fdfd
    from opencl_fdfd import csr
from fdfd_tools.solvers import generic as generic_solver
import scipy.io as sio
import fdfd_tools
from fdfd_tools import vec, unvec, waveguide_mode
import numpy
from typing import List, Tuple
import scipy.sparse as sparse
from collections import Iterable
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'

print('here we go')

#usage: simulation.py jobName.mat plotMe
jobName = sys.argv[1]
plotMe = sys.argv[2]

if plotMe == '1':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

__author__ = 'Marco Butz'

mat = sio.loadmat(jobName, squeeze_me=True)

#all input arrays have to be 1D:
#epsilonCell = mat.epsilon
#x = mat.epsilon[0,0]
#y = mat.epsilon[0,1]
#z = mat.epsilon[0,2]
inputEpsilon = fdfd_tools.vec(f=[mat['epsilon'],mat['epsilon'],mat['epsilon']])

omega = mat['omega']
print('omega: ', omega)
#omega is needed as 2*pi/lambda
pixelSize = mat['pixelSize']
for i in range(0,3):
    print("mat['dims'][",i,"]: ", mat['dims'][i])
#epsilon = numpy.hstack(tuple((fi.flatten(order='F') for fi in [mat['epsilon'][0,0], mat['epsilon'][0,1], mat['epsilon'][0,2]])))
#J = numpy.hstack(tuple((fi.flatten(order='F') for fi in [mat['bufJ'][0,0], mat['bufJ'][0,1], mat['bufJ'][0,2]])))

dxIsotrop_x = numpy.full(mat['dims'][0], pixelSize).astype(float)
dxIsotrop_y = numpy.full(mat['dims'][1], pixelSize).astype(float)
dxIsotrop_z = numpy.full(mat['dims'][2], pixelSize).astype(float)

dims = mat['dims']
print('shape of eps: ', inputEpsilon.shape)
#generate_periodic_dx?!?!
dxes = [[dxIsotrop_x,dxIsotrop_y,dxIsotrop_z],[dxIsotrop_x,dxIsotrop_y,dxIsotrop_z]]
#dxes = fdfd_tools.grid.generate_periodic_dx([dxIsotrop_x,dxIsotrop_y,dxIsotrop_z])
print('shape of dxes (x): ', dxIsotrop_x.shape)
print('shape of dxes (y): ', dxIsotrop_y.shape)
print('shape of dxes (z): ', dxIsotrop_z.shape)
#create PMLs
if mat['dims'][2] == 1:
    print('applying two pml axes')
    pml_axes = (0, 1)
    mat['epsilon'] = numpy.expand_dims(mat['epsilon'], axis=2)
    pml_thickness = [10,10,0]
else:
    pml_axes = (0, 1, 2)
    pml_thickness = [10,10,10]

eff_eps = [[1.0,1.0],[1.0,1.0],[2.08,1.0]]
for a in pml_axes:
    for p in (-1, 1):
        if p==-1:
            eff_eps_i = 0
        else:
            eff_eps_i = 1
        dxes = fdfd_tools.grid.stretch_with_scpml(dxes, axis=a, polarity=p, omega=omega,
                                                  thickness=pml_thickness[a], epsilon_effective=eff_eps[a][eff_eps_i])


#print(dxes[0][2][1])
#solve waveguide modes:
#define mode slice
#pos is [[x, y, z], [x, y, z]]
modeSourcePos = mat['modeSourcePos']
slices = (modeSourcePos[0].astype(int),
            modeSourcePos[1].astype(int))

for j in mat['modeSourcePos'][0].astype(int):
    print(j)
for j in mat['modeSourcePos'][1].astype(int):
    print(j)

bufferEps = [mat['epsilon'] for _ in range(3)]
print(mat['epsilon'].shape)
wgFormatedEps = numpy.stack(bufferEps, axis=0)
print(wgFormatedEps.shape)

modeArgs = {
    'omega': omega,
    'slices': [slice(i, f+1) for i, f in zip(*slices)],
    'dxes': dxes,
    'axis': 0,
    'polarity': +1,
}

source = fdfd_tools.waveguide_mode.solve_waveguide_mode(mode_number=mat['modeSourceNum'],
                         **modeArgs, #f+1 because second param of slice is NOT included
                         epsilon=wgFormatedEps
                         )
#calculate self overlap
modeselfOverlap = waveguide_mode.compute_overlap_e(**modeArgs, **source)
print('source[E].shape: ', source['E'][0].shape, ' ', source['E'][1].shape, ' ',source['E'][2].shape)
print('source self overlap: ', numpy.abs(vec(source['E']) @ vec(modeselfOverlap)).astype(float))

modeJ = fdfd_tools.waveguide_mode.compute_source(**modeArgs, **source)
print('solved source')
modesToMeasure = [dict() for _ in range(mat['numModesToMeasure'])]

overlapModesToMeasure = []
#the following two are first use to wrap stuff in an array if mat file only contains one element and the array got squeezed
#they are later being cleared again and used otherwise
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

print('will measure ', mat['numModesToMeasure'], ' modes')
print('length of posModesToMeasure: ', len(posModesToMeasure))
for i in range(0,mat['numModesToMeasure']):
    #define mode slice
    #pos is [[x, y, z], [x, y, z]]
    slices = (posModesToMeasure[i][0].astype(int),
                posModesToMeasure[i][1].astype(int))


    for j in posModesToMeasure[i][0].astype(int):
        print(j)
    for j in posModesToMeasure[i][1].astype(int):
        print(j)

    modeArgs = {
        'omega': omega,
        'slices': [slice(i, f+1) for i, f in zip(*slices)],
        'dxes': dxes,
        'axis': 0,
        'polarity': +1,
    }
    print('will prepare measuring the ', modeNumModesToMeasure[i], ' mode')
    modeResult = fdfd_tools.waveguide_mode.solve_waveguide_mode(mode_number=modeNumModesToMeasure[i],
                             **modeArgs,
                             epsilon=wgFormatedEps
                             )
    modeOverlap = waveguide_mode.compute_overlap_e(**modeArgs, **modeResult)

    modesToMeasure[i].update({'modeNum': modeNumModesToMeasure[i], 'pos': posModesToMeasure[i], 'modeSourcePos': mat['modeSourcePos'], 'E': modeResult['E'], 'slices': slices, 'overlap_operator': modeOverlap})

posModesToMeasure = []
modeNumModesToMeasure = []

start = time.time()

if module_available('opencl_fdfd') and mat['dims'][2] != 1:
    E = fdfd.cg_solver(
        omega=omega,
        dxes=dxes,
        J=fdfd_tools.vectorization.vec(modeJ),
        epsilon=inputEpsilon,
        adjoint=False,
        max_iters=7000,
        err_threshold=1e-6)
else:
    print('warning: using generic_solver!')
    E = generic_solver(
        omega=omega,
        dxes=dxes,
        J=fdfd_tools.vectorization.vec(modeJ),
        epsilon=inputEpsilon,
        adjoint=False,
        matrix_solver=scipy.sparse.linalg.spsolve)
    """
    #this is the sparse matrix solver
    fdfd_args = {
        'omega':omega,
        'dxes':dxes,
        'J':fdfd_tools.vectorization.vec(modeJ),
        'epsilon':inputEpsilon,
        'adjoint':False,
    }
    """
    #E = csr.fdfd_cg_solver(**fdfd_args)


end = time.time()
print('simulation took: ', str(end - start), 's')

#aquire H with
H = fdfd_tools.operators.e2h(omega, dxes) @ E

#calculate overlaps
for mode in modesToMeasure:
    mode.update({'overlap': numpy.abs(vec(E) @ vec(mode['overlap_operator'])).astype(float)})
    posModesToMeasure.append(mode['pos'])
    modeNumModesToMeasure.append(mode['modeNum'])
    overlapModesToMeasure.append(mode['overlap'])

print("shape of resulting E: ", E.shape)
shape = numpy.ndarray(shape=(mat['dims'][0], mat['dims'][1], mat['dims'][2]))
print("unvectorizing into shape: ", shape.shape)
E3DCellArray = fdfd_tools.vectorization.unvec(v=E, shape=shape.shape)
H3DCellArray = fdfd_tools.vectorization.unvec(v=H, shape=shape.shape)

if plotMe == '1':
    print('will plot now')
    print('plotting ', int(dims[2]/2))
    def pcolor(v):
        vmax = numpy.max(numpy.abs(v))
        pyplot.pcolor(v, cmap='seismic', vmin=-vmax, vmax=vmax)
        pyplot.axis('equal')
        pyplot.colorbar()
    pyplot.figure()
    pcolor(numpy.real(E3DCellArray[2][:, :, int(dims[2]/2)]))
    pyplot.savefig('debug-ez-field-' + str(mat['modeSourceNum']) + '.png', dpi=300)
    pyplot.figure()
    pcolor(numpy.real(mat['epsilon'][:, :, int(dims[2]/2)]))
    pyplot.savefig('debug-structure-' + str(mat['modeSourceNum']) + '.png', dpi=300)
    pyplot.figure()
    pcolor(numpy.real(source['E'][2][:, :, int(dims[2]/2)]))
    pyplot.savefig('debug-ez-source-' + str(mat['modeSourceNum']) + '.png', dpi=300)

    #debug txt structure
    numpy.savetxt('debug-structure-' + str(mat['modeSourceNum']) + '.txt',
                    numpy.real(mat['epsilon'][:, :, int(dims[2]/2)]), fmt='%.0u', delimiter='')
    #pyplot.subplot(2, 2, 3)
    #pcolor(numpy.real(modeJ[1][12, :, :]))
    #pyplot.show()

result = {'E': E3DCellArray, 'H': H3DCellArray}
print("length of E3DCellArray: ", len(E3DCellArray))
print("shape of unvectorized field: ", E3DCellArray[0].shape, E3DCellArray[1].shape, E3DCellArray[2].shape)
FrameStackE = numpy.empty((len(E3DCellArray),), dtype=numpy.object)
FrameStackH = numpy.empty((len(H3DCellArray),), dtype=numpy.object)
for i in range(len(E3DCellArray)):
    print("is any of E3DCellArray",i," different from 0?", numpy.any(E3DCellArray[i]))
    FrameStackE[i] = E3DCellArray[i]
    FrameStackH[i] = H3DCellArray[i]
jobNameWithoutPath = jobName.split('/')[len(jobName.split('/'))-1]
sio.savemat("results_" + jobNameWithoutPath, {'pos': posModesToMeasure,
                                    'modeNum': modeNumModesToMeasure,
                                    'overlap': overlapModesToMeasure,
                                    'inputModeNum': mat['modeSourceNum'],
                                    'inputModePos': mat['modeSourcePos']})

#delete old job data
#subprocess.call(['rm', jobName])
