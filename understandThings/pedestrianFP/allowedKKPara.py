import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad
import math
import scipy.optimize

import matplotlib as mpl
from matplotlib import gridspec
import scipy.constants as consts

fontsize = 8

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 8
mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage[helvet]{sfmath}',
]



def rootFuncTE(k, kparaSq, d, eps):
    kD = np.sqrt((eps - 1) * kparaSq + eps * k**2)
    term1 = k * np.cos(k * d / 2.) * np.sin(kD * d / 2.)
    term2 = kD * np.cos(kD * d / 2.) * np.sin(k * d / 2.)
    return  term1 + term2

def rootFuncTEEva(k, kparaSq, d, eps):
    kD = np.sqrt((eps - 1) * kparaSq + eps * k**2)
    term1 = kD * np.tanh(k * d / 2) * np.cos(kD * d / 2)
    term2 = k * np.sin(kD * d / 2)
    return term1 + term2

def extremalPoints(d, omega, kparaSq, eps, N, mode):
    rootFunc = rootFuncTE
    if(mode == "TEEva"):
        rootFunc = rootFuncTEEva

    upperBound = omega / consts.c
    if(mode == "TEEva"):
        upperBound = np.sqrt(eps - 1.) * omega / consts.c

    intervals = np.zeros((N, 2))
    lowerBounds = np.linspace(0, upperBound, N, endpoint=False)
    upperBounds = np.linspace(0, upperBound, N, endpoint=False) + lowerBounds[1]
    intervals[:, 0] = lowerBounds
    intervals[:, 1] = upperBounds
    if(mode == "TEEva"):
        print("upperBound = {}".format(upperBound))
        print("upperInterval = {}".format(intervals[-1, 1]))
    extrema = np.zeros(0)
    prevMaxatMaxInd = False#something to include maxima that are on boundaries of intervals
    for n in range(N):
        Ntemp = 3
        tempArr = np.linspace(intervals[n, 0], intervals[n, 1], Ntemp)
        maxInd = np.argmax(rootFunc(tempArr, kparaSq, d, eps) ** 2)
        if(maxInd == Ntemp - 1):
            prevMaxatMaxInd = True
            continue
        if(maxInd == 0 and not(prevMaxatMaxInd)):
            prevMaxatMaxInd = False
            continue
        extrema = np.append(extrema, tempArr[maxInd])
        prevMaxatMaxInd = False
    return extrema

def computeRootsGivenExtrema(d, omega, kparaSq, eps, extrema, mode):
    rootFunc = rootFuncTE
    if(mode == "TEEva"):
        rootFunc = rootFuncTEEva

    upperBound = omega / consts.c
    if(mode == "TEEva"):
        upperBound = np.sqrt(eps - 1.) * omega / consts.c

    nRoots = len(extrema)
    roots = np.zeros(nRoots)
    intervals = np.zeros((nRoots, 2))
    intervals[0, :] = np.array([0, extrema[0]])
    intervals[1:, 0] = extrema[:-1]
    intervals[1:, 1] = extrema[1:]
    if(rootFunc(extrema[-1], kparaSq, d, eps) * rootFunc(upperBound - 1e-6, kparaSq, d, eps) < 0):
        intervals = np.append(intervals, np.array([[extrema[-1], upperBound - 1e-6]]), axis = 0)
    for rootInd, root in enumerate(roots):
        #not always a root between two adjacent extrema
        #if(rootFunc(intervals[rootInd, 0], L, omega, eps) * rootFunc(intervals[rootInd, 1], L, omega, eps) > 0):
        #    continue
        tempRoot = scipy.optimize.root_scalar(rootFunc, args = (kparaSq, d, eps), bracket=tuple(intervals[rootInd, :]))
        roots[rootInd] = tempRoot.root
    if(rootFunc(extrema[-1], kparaSq, d, eps) * rootFunc(upperBound, kparaSq, d, eps) < 0):
        lastRoot = scipy.optimize.root_scalar(rootFunc, args = (kparaSq, d, eps), bracket=(extrema[-1], upperBound))
        roots = np.append(roots, lastRoot.root)
    if(roots[0] == 0.):
        return roots[1:]

    return roots


def plotRootFuncWithRoots(d, omega, kparaSq, eps, roots):

    upperBound = omega / consts.c

    kArr = np.linspace(0., upperBound, 1000)
    rootFuncVals = rootFuncTE(kArr, kparaSq, d, eps)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    for exInd, roots in enumerate(roots):
        ax.axvline(roots * consts.c / omega, color ='gray', lw = 0.4)

    ax.plot(kArr * consts.c / omega, rootFuncVals, color='indianred', lw=0.8)
    ax.set_xlim(np.amin(kArr) * consts.c / omega, np.amax(kArr) * consts.c / omega)

    ax.axhline(0., color = 'gray', lw = 0.5)
    ax.axvline(1., color = 'teal', lw = 1.)

    ax.set_xlabel(r'$k_z \, [\frac{\omega}{c}]$')
    ax.set_ylabel(r'$K(\omega)$')

    plt.savefig("./savePedestrianFPPlots/rootFuncWithRootsKPara.png")


def findKs(d, omega, kx, ky, eps, mode):
    kparaSq = kx**2 + ky**2
    NDiscrete = 11 * int(omega / consts.c * (np.sqrt(eps) + 1.) * d + 17)
    extrema = extremalPoints(d, omega, kparaSq, eps, NDiscrete, mode)
    if(len(extrema) == 0):
        return np.array([])
    roots = computeRootsGivenExtrema(d, omega, kparaSq, eps, extrema, mode)
    return roots

def findKsKPara(d, omega, kparaSq, eps, mode):
    NDiscrete = 11 * int(omega / consts.c * (np.sqrt(eps) + 1.) * d + 17)
    extrema = extremalPoints(d, omega, kparaSq, eps, NDiscrete, mode)
    roots = computeRootsGivenExtrema(d, omega, kparaSq, eps, extrema, mode)
    return roots

