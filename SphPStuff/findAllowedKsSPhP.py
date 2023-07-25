import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import h5py
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import complexIntegral
import scipy.constants as consts
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy

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
    #    r'\renewcommand{\familydefault}{\sfdefault}',
    #    r'\usepackage[scaled=1]{helvet}',
    r'\usepackage[helvet]{sfmath}',
    #    r'\everymath={\sf}'
]

import epsilonFunctions as epsFunc

def rootFuncTE(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromK(k, omega, wLO, wTO, epsInf)
    term1 = k * np.cos(k * L / 2.) * np.sin(kD * L / 2.)
    term2 = kD * np.cos(kD * L / 2.) * np.sin(k * L / 2.)
    return  term1 + term2

def rootFuncTEEva(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKEva(k, omega, wLO, wTO, epsInf)
    term1 = k / kD * np.sin(kD * L / 2)
    term2 = np.cos(kD * L / 2) * np.tanh(k * L / 2)
    return term1 + term2

def rootFuncTERes(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKRes(k, omega, wLO, wTO, epsInf)
    term1 = k * np.cos(k * L / 2) * np.tanh(kD * L / 2)
    term2 = kD * np.sin(k * L / 2)
    return term1 + term2

def rootFuncTM(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromK(k, omega,wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k / kD * np.sin(k * L / 2.) * np.cos(kD * L / 2.)
    term2 = np.cos(k * L / 2.) * np.sin(kD * L / 2)
    return term1 + term2

def rootFuncTMEva(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKEva(k, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k * np.tanh(k * L / 2) * np.cos(kD * L / 2)
    term2 = kD * np.sin(kD * L / 2)
    return term1 - term2

def rootFuncTMRes(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKRes(k, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k / kD * np.sin(k * L / 2)
    term2 = np.cos(k * L / 2) * np.tanh(kD * L / 2)
    return term1 - term2

def rootFuncSurf(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKSurf(k, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k / kD * np.tanh(k * L / 2)
    term2 = np.tanh(kD * L / 2)
    return term1 + term2

def getUpperBound(mode, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    upperBound = 0
    if (mode == "TE" or mode == "TM" or mode == "Surf"):
            upperBound = omega / consts.c
    elif (mode == "TEEva" or mode == "TMEva"):
        upperBound = np.sqrt(eps - 1) * omega / consts.c - 1e-5
    elif (mode == "TERes" or mode == "TMRes"):
        if (eps < 0):
            upperBound = omega / consts.c
        elif(eps < 1 and eps > 0):
            upperBound = np.sqrt(1 - eps) * omega / consts.c - 1e-5

    return upperBound

def getLowerBound(mode, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    lowerBound = 0
    if(eps > 0 and eps < 1):
        if (mode == "TE" or mode == "TM"):
            lowerBound = np.sqrt(1 - eps) * omega / consts.c + 1e-5

    return lowerBound

def extremalPoints(L, omega, wLO, wTO, epsInf, N, mode):
    rootFunc = rootFuncTE
    if(mode == "TE"):
        rootFunc = rootFuncTE
    elif (mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva
    elif (mode == "TERes"):
        rootFunc = rootFuncTERes
    elif (mode == "TMRes"):
        rootFunc = rootFuncTMRes
    elif (mode == "Surf"):
        rootFunc = rootFuncSurf
    else:
        print("Error: specified mode doesn't exist!!!!!!!!!!!!!!!!")
        exit()

    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    intervals = np.zeros((N, 2))
    lowerBounds = np.linspace(lowerBound, upperBound, N, endpoint=False)
    upperBounds = np.linspace(lowerBound, upperBound, N, endpoint=False) + (lowerBounds[1] - lowerBounds[0])
    intervals[:, 0] = lowerBounds
    intervals[:, 1] = upperBounds
    extrema = np.zeros(0)
    prevMaxatMaxInd = False#something to include maxima that are on boundaries of intervals
    for n in range(N):
        Ntemp = 3
        tempArr = np.linspace(intervals[n, 0], intervals[n, 1], Ntemp)
        maxInd = np.argmax(rootFunc(tempArr, L, omega, wLO, wTO, epsInf) ** 2)
        if(maxInd == Ntemp - 1):
            prevMaxatMaxInd = True
            continue
        if(maxInd == 0 and not(prevMaxatMaxInd)):
            prevMaxatMaxInd = False
            continue
        extrema = np.append(extrema, tempArr[maxInd])
        prevMaxatMaxInd = False
    return extrema

def computeRootsGivenExtrema(L, omega, wLO, wTO, epsInf, extrema, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva
    elif (mode == "TERes"):
        rootFunc = rootFuncTERes
    elif (mode == "TMRes"):
        rootFunc = rootFuncTMRes
    elif (mode == "Surf"):
        rootFunc = rootFuncSurf

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    nRoots = len(extrema)
    roots = np.zeros(nRoots)
    intervals = np.zeros((nRoots, 2))
    intervals[0, :] = np.array([lowerBound, extrema[0]])
    intervals[1:, 0] = extrema[:-1]
    intervals[1:, 1] = extrema[1:]
    if(rootFunc(extrema[-1], L, omega, wLO, wTO, epsInf) * rootFunc(upperBound - 1e-6, L, omega, wLO, wTO, epsInf) < 0):
        intervals = np.append(intervals, np.array([[extrema[-1], upperBound - 1e-6]]), axis = 0)
    for rootInd, root in enumerate(roots):
        #not always a root between two adjacent extrema
        #if(rootInd == 0 and rootFunc(intervals[rootInd, 0], L, omega, wLO, wTO, epsInf) * rootFunc(intervals[rootInd, 1], L, omega, wLO, wTO, epsInf) > 0):
        if (rootFunc(intervals[rootInd, 0], L, omega, wLO, wTO, epsInf) * rootFunc(
                    intervals[rootInd, 1], L, omega, wLO, wTO, epsInf) > 0):
                continue
        tempRoot = scipy.optimize.root_scalar(rootFunc, args = (L, omega, wLO, wTO, epsInf), bracket=tuple(intervals[rootInd, :]))
        roots[rootInd] = tempRoot.root

    if(roots[0] == 0.):
        return roots[1:]

    return roots

def plotRootFuncWithExtrema(L, omega, wLO, wTO, epsInf, extrema, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva
    elif (mode == "TERes"):
        rootFunc = rootFuncTERes
    elif (mode == "TMRes"):
        rootFunc = rootFuncTMRes
    elif (mode == "Surf"):
        rootFunc = rootFuncSurf

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    kArr = np.linspace(lowerBound, upperBound - 1e-6, 1000)
    rootFuncVals = rootFunc(kArr, L, omega, wLO, wTO, epsInf)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    for exInd, extrema in enumerate(extrema):
        ax.axvline(extrema, color = 'gray', lw = 0.5)

    ax.plot(kArr, rootFuncVals, color='indianred', lw=0.8)
    #ax.set_xlim(np.amin(kArr), np.amax(kArr))
    #ax.set_xlim(np.amax(kArr) * 0.9, np.amax(kArr))


    ax.axhline(0., color = 'gray', lw = 0.5)
    ax.axvline(omega / consts.c, color = 'teal', lw = 0.8)

    plt.savefig("./SPhPPlotsSaved/rootFuncWithExtrema" + mode + ".png")

def plotRootFuncWithRoots(L, omega, wLO, wTO, epsInf, roots, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva
    elif (mode == "TERes"):
        rootFunc = rootFuncTERes
    elif (mode == "TMRes"):
        rootFunc = rootFuncTMRes
    elif (mode == "Surf"):
        rootFunc = rootFuncSurf

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    kArr = np.linspace(0., upperBound - 1e-6, 1000)
    rootFuncVals = rootFunc(kArr, L, omega, wLO, wTO, epsInf)
    #rootFuncValsEps1 = rootFunc(kArr, L, omega, 1., c)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    for exInd, roots in enumerate(roots):
        ax.axvline(roots * consts.c / omega, color ='gray', lw = 0.4)

    ax.plot(kArr * consts.c / omega, rootFuncVals, color='indianred', lw=0.8)
    #ax.plot(kArr, rootFuncValsEps1, color='teal', lw=0.8, linestyle = '--')
    ax.set_xlim(np.amin(kArr) * consts.c / omega, np.amax(kArr) * consts.c / omega)

    ax.axhline(0., color = 'gray', lw = 0.5)
    ax.axvline(1., color = 'teal', lw = 1.)

    ax.set_xlabel(r'$k_z \, [\frac{\omega}{c}]$')
    ax.set_ylabel(r'$K(\omega)$')

    plt.savefig("./SPhPPlotsSaved/rootFuncWithRootsSPhP" + mode + ".png")

def allowedKSurf(L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    root = scipy.optimize.root_scalar(rootFuncSurf, args=(L, omega, wLO, wTO, epsInf),
                                          bracket=tuple([0., omega / consts.c * (10 + 10 / (epsAbs - 1))]))
    return np.array([root.root])

def computeAllowedKs(L, omega, wLO, wTO, epsInf, mode):
    if(mode == "Surf"):
        return allowedKSurf(L, omega, wLO, wTO, epsInf)
    else:
        eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
        NDiscrete = 11 * int(omega / consts.c * (np.sqrt(np.abs(eps)) + 1.) * L  + 17)
        print("NDiscrete = {}".format(NDiscrete))
        extrema = extremalPoints(L, omega, wLO, wTO, epsInf, NDiscrete, mode)
        if(len(extrema) == 0):
            return np.array([])
        #plotRootFuncWithExtrema(L, omega, wLO, wTO, epsInf, extremaTE, mode)
        roots = computeRootsGivenExtrema(L, omega, wLO, wTO, epsInf, extrema, mode)
        print("Number of roots for " + mode +" found = {}".format(roots.shape))
        plotRootFuncWithRoots(L, omega, wLO, wTO, epsInf, roots, mode)
        print("roots = {}".format(roots))
        return roots

def buildExtremaFromRoots(roots, omega, wLO, wTO, epsInf, mode):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    rootsBelow = np.append([lowerBound], roots)
    rootsAbove = np.append(roots, [upperBound])
    extrema = (rootsBelow + rootsAbove) / 2.
    return extrema

def findKsDerivativeW(roots, L, omega, wLO, wTO, epsInf, mode):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    delOm = omega * 1e-7
    extrema = buildExtremaFromRoots(roots, omega, wLO, wTO, epsInf, mode)
    NDiscrete = 51 * int(omega / consts.c * (np.sqrt(np.abs(eps)) + 1.) * L + 17)
    #extrema = extremalPoints(L, omega, wLO, wTO, epsInf, NDiscrete, mode)
    if (len(extrema) == 0):
        return np.array([])
    rootsPlus = computeRootsGivenExtrema(L, omega + delOm, wLO, wTO, epsInf, extrema, mode)
    rootsPlus = rootsPlus[:len(roots)]
    print("rootsPlus = {}".format(rootsPlus))
    print("extrema = {}".format(extrema))
    return (rootsPlus - roots) / (delOm)

def findKsSurf(L, omega, wLO, wTO, epsInf):
    kVals = allowedKSurf(L, omega, wLO, wTO, epsInf)
    if (len(kVals) == 0):
        return 0
    else:
        return kVals[0]

def findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf):
    delOm = omega * 1e-4
    rootPlus = allowedKSurf(L, omega + delOm, wLO, wTO, epsInf)[0]
    rootMinus = allowedKSurf(L, omega - delOm, wLO, wTO, epsInf)[0]

    return (rootPlus - rootMinus) / (2. * delOm)

