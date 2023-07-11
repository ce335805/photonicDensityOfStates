import numpy as np
import scipy.constants as consts
import scipy.optimize


def rootFuncTE(k, kparaSq, L, eps):
    kD = np.sqrt((eps - 1) * kparaSq + eps * k**2)
    term1 = k * np.cos(k * L / 2.) * np.sin(kD * L / 2.)
    term2 = kD * np.cos(kD * L / 2.) * np.sin(k * L / 2.)
    return  term1 + term2

def rootFuncTEEva(k, kparaSq, L, eps):
    kD = np.sqrt((eps - 1) * kparaSq + eps * k**2)
    term1 = kD * np.tanh(k * L / 2) * np.cos(kD * L / 2)
    term2 = k * np.sin(kD * L / 2)
    return term1 + term2

def extremalPoints(L, omega, kparaSq, eps, N, mode):
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
        maxInd = np.argmax(rootFunc(tempArr, kparaSq, L, eps) ** 2)
        if(maxInd == Ntemp - 1):
            prevMaxatMaxInd = True
            continue
        if(maxInd == 0 and not(prevMaxatMaxInd)):
            prevMaxatMaxInd = False
            continue
        extrema = np.append(extrema, tempArr[maxInd])
        prevMaxatMaxInd = False
    return extrema

def computeRootsGivenExtrema(L, omega, kparaSq, eps, extrema, mode):
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
    if(rootFunc(extrema[-1], kparaSq, L, eps) * rootFunc(upperBound - 1e-6, kparaSq, L, eps) < 0):
        intervals = np.append(intervals, np.array([[extrema[-1], upperBound - 1e-6]]), axis = 0)
    for rootInd, root in enumerate(roots):
        #not always a root between two adjacent extrema
        #if(rootFunc(intervals[rootInd, 0], L, omega, eps) * rootFunc(intervals[rootInd, 1], L, omega, eps) > 0):
        #    continue
        tempRoot = scipy.optimize.root_scalar(rootFunc, args = (kparaSq, L, eps), bracket=tuple(intervals[rootInd, :]))
        roots[rootInd] = tempRoot.root

    if(roots[0] == 0.):
        return roots[1:]

    return roots

def findKs(L,omega, kx, ky, eps, mode):
    kparaSq = kx**2 + ky**2
    NDiscrete = 11 * int(omega / consts.c * (np.sqrt(eps) + 1.) * L + 17)
    extremaTEEva = extremalPoints(L, omega, kparaSq, eps, NDiscrete, mode)
    rootsTEEva = computeRootsGivenExtrema(L, omega, kparaSq, eps, extremaTEEva, mode)
    return rootsTEEva

def findKsKPara(L,omega, kparaSq, eps, mode):
    NDiscrete = 11 * int(omega / consts.c * (np.sqrt(eps) + 1.) * L + 17)
    extremaTEEva = extremalPoints(L, omega, kparaSq, eps, NDiscrete, mode)
    rootsTEEva = computeRootsGivenExtrema(L, omega, kparaSq, eps, extremaTEEva, mode)
    return rootsTEEva

def findKsKParaDerivative(L,omega, kparaSq, eps, mode):
    deltaKParaSq = kparaSq * 1e-4
    NDiscrete = 11 * int(omega / consts.c * (np.sqrt(eps) + 1.) * L + 17)
    extrema = extremalPoints(L, omega, kparaSq, eps, NDiscrete, mode)
    rootsPlus = computeRootsGivenExtrema(L, omega, kparaSq + deltaKParaSq, eps, extrema, mode)
    rootsMinus = computeRootsGivenExtrema(L, omega, kparaSq - deltaKParaSq, eps, extrema, mode)
    return (rootsPlus - rootsMinus) / (2. * deltaKParaSq) * 2. * np.sqrt(kparaSq)


