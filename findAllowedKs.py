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


def rootFuncTE(k, L, omega, eps):
    c = consts.c
    kD = np.sqrt((eps - 1) * omega**2 / c**2 + k**2)
    term1 = k * np.cos(k * L / 2.) * np.sin(kD * L / 2.)
    term2 = kD * np.cos(kD * L / 2.) * np.sin(k * L / 2.)
    return  term1 + term2

def rootFuncTM(k, L, omega, eps):
    c = consts.c
    kD = np.sqrt((eps - 1) * omega**2 / c**2 + k**2)
    term1 = eps * k * np.sin(k * L / 2.) * np.cos(kD * L / 2.)
    term2 = kD * np.cos(k * L / 2.) * np.sin(kD * L / 2)
    return term1 + term2

def rootFuncTEEva(kD, L, omega, eps):
    kReal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kD ** 2 + 1e-8)
    term1 = kD * np.tanh(kReal * L / 2) * np.cos(kD * L / 2)
    term2 = kReal * np.sin(kD * L / 2)
    return term1 + term2


def rootFuncTMEva(kD, L, omega, eps):
    kReal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kD ** 2 + 1e-6)
    term1 = eps * kReal * np.tanh(kReal * L / 2) * np.cos(kD * L / 2)
    term2 = kD * np.sin(kD * L / 2)
    return term1 - term2

def extremalPoints(L, omega, eps, N, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva

    upperBound = 0
    if(mode == "TE" or mode == "TM"):
        upperBound = omega / consts.c
    elif(mode == "TEEva" or mode == "TMEva"):
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
        maxInd = np.argmax(rootFunc(tempArr, L, omega, eps) ** 2)
        if(maxInd == Ntemp - 1):
            prevMaxatMaxInd = True
            continue
        if(maxInd == 0 and not(prevMaxatMaxInd)):
            prevMaxatMaxInd = False
            continue
        extrema = np.append(extrema, tempArr[maxInd])
        prevMaxatMaxInd = False
    return extrema

def computeRootsGivenExtrema(L, omega, eps, extrema, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva

    upperBound = 0
    if(mode == "TE" or mode == "TM"):
        upperBound = omega / consts.c
    elif(mode == "TEEva" or mode == "TMEva"):
        upperBound = np.sqrt(eps - 1.) * omega / consts.c

    nRoots = len(extrema)
    roots = np.zeros(nRoots)
    intervals = np.zeros((nRoots, 2))
    intervals[0, :] = np.array([0, extrema[0]])
    intervals[1:, 0] = extrema[:-1]
    intervals[1:, 1] = extrema[1:]
    if(rootFunc(extrema[-1], L, omega, eps) * rootFunc(upperBound - 1e-6, L, omega, eps) < 0):
        intervals = np.append(intervals, np.array([[extrema[-1], upperBound - 1e-6]]), axis = 0)
    for rootInd, root in enumerate(roots):
        #not always a root between two adjacent extrema
        #if(rootFunc(intervals[rootInd, 0], L, omega, eps) * rootFunc(intervals[rootInd, 1], L, omega, eps) > 0):
        #    continue
        tempRoot = scipy.optimize.root_scalar(rootFunc, args = (L, omega, eps), bracket=tuple(intervals[rootInd, :]))
        roots[rootInd] = tempRoot.root

    if(roots[0] == 0.):
        return roots[1:]

    return roots

def plotRootFuncWithExtrema(L, omega, eps, extrema, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva


    upperBound = 0
    if(mode == "TE" or mode == "TM"):
        upperBound = omega / consts.c
    elif(mode == "TEEva" or mode == "TMEva"):
        upperBound = np.sqrt(eps - 1.) * omega / consts.c

    c = consts.c

    kArr = np.linspace(0., upperBound - 1e-6, 1000)
    rootFuncVals = rootFunc(kArr, L, omega, eps)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    for exInd, extrema in enumerate(extrema):
        ax.axvline(extrema, color = 'gray', lw = 0.5)

    ax.plot(kArr, rootFuncVals, color='indianred', lw=0.8)
    ax.set_xlim(np.amin(kArr), np.amax(kArr))


    ax.axhline(0., color = 'gray', lw = 0.5)
    ax.axvline(omega / c, color = 'teal', lw = 0.8)

    plt.savefig("./savedPlots/rootFuncWithExtrema" + mode + ".png")

def plotRootFuncWithRoots(L, omega, eps, roots, mode):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva

    upperBound = 0
    if(mode == "TE" or mode == "TM"):
        upperBound = omega / consts.c
    elif(mode == "TEEva" or mode == "TMEva"):
        upperBound = np.sqrt(eps - 1.) * omega / consts.c

    c = consts.c

    print("omega / c = {}".format(omega / c))

    kArr = np.linspace(0., upperBound - 1e-6, 1000)
    rootFuncVals = rootFunc(kArr, L, omega, eps)
    #rootFuncValsEps1 = rootFunc(kArr, L, omega, 1., c)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    for exInd, roots in enumerate(roots):
        ax.axvline(roots * c / omega, color ='gray', lw = 0.4)

    ax.plot(kArr * c / omega, rootFuncVals, color='indianred', lw=0.8)
    #ax.plot(kArr, rootFuncValsEps1, color='teal', lw=0.8, linestyle = '--')
    ax.set_xlim(np.amin(kArr) * c / omega, np.amax(kArr) * c / omega)


    ax.axhline(0., color = 'gray', lw = 0.5)
    ax.axvline(1., color = 'teal', lw = 1.)

    ax.set_xlabel(r'$k_z \, [\frac{\omega}{c}]$')
    ax.set_ylabel(r'$K(\omega)$')

    plt.savefig("./savedPlots/rootFuncWithRoots" + mode + ".png")
