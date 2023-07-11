import numpy as np
import scipy.constants as consts
import scipy.optimize as opt
import scipy


def rootFuncTE(k, L, omega, eps):
    kD = np.sqrt((eps - 1) * omega**2 / consts.c**2 + k**2)
    term1 = k * np.cos(k * L / 2.) * np.sin(kD * L / 2.)
    term2 = kD * np.cos(kD * L / 2.) * np.sin(k * L / 2.)
    return  term1 + term2

def rootFuncTEEva(kD, L, omega, eps):
    kReal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kD ** 2 + 1e-8)
    term1 = kD * np.tanh(kReal * L / 2) * np.cos(kD * L / 2)
    term2 = kReal * np.sin(kD * L / 2)
    return term1 + term2

def extremalPoints(L, omega, eps, N, mode):
    rootFunc = rootFuncTE
    if (mode == "TEEva"):
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


def findKs(L,omega, eps, mode):
    NDiscrete = 51 * int(omega / consts.c * (np.sqrt(eps) + 1.) * L + 17)
    extrema = extremalPoints(L, omega, eps, NDiscrete, mode)
    if(len(extrema) == 0):
        return np.array([])
    roots = computeRootsGivenExtrema(L, omega, eps, extrema, mode)
    return roots

def findKsDerivativeW(L,omega, eps, mode):
    delOm = omega * 1e-4
    NDiscrete = 51 * int(omega / consts.c * (np.sqrt(eps) + 1.) * L + 17)
    extrema = extremalPoints(L, omega, eps, NDiscrete, mode)
    rootsPlus = computeRootsGivenExtrema(L, omega + delOm, eps, extrema, mode)
    #print("rootsPlus = {}".format(rootsPlus))
    rootsMinus = computeRootsGivenExtrema(L, omega - delOm, eps, extrema, mode)
    #print("rootsMinus = {}".format(rootsMinus))
    return (rootsPlus - rootsMinus) / (2. * delOm)