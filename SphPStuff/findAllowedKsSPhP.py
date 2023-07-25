import numpy as np
import scipy.constants as consts
import scipy.optimize as opt
import scipy

import epsilonFunctions as epsFunc
import plotRootFuncs

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


def allowedKSurf(L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    root = scipy.optimize.root_scalar(rootFuncSurf, args=(L, omega, wLO, wTO, epsInf),
                                          bracket=tuple([0., omega / consts.c * (10 + 10 / (epsAbs - 1))]))
    return np.array([root.root])

def createRootsFuncPlotWithLines(lines, L, omega, wLO, wTO, epsInf, mode, nameAdd):
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

    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    kArr = np.linspace(lowerBound, upperBound, 1000)
    rootFuncVals = rootFunc(kArr, L, omega, wLO, wTO, epsInf)

    plotRootFuncs.plotRootFuncWithRoots(kArr, rootFuncVals, lines, omega, mode, nameAdd)

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
        createRootsFuncPlotWithLines(extrema, L, omega, wLO, wTO, epsInf, mode, "Extrema")
        roots = computeRootsGivenExtrema(L, omega, wLO, wTO, epsInf, extrema, mode)
        print("Number of roots for " + mode +" found = {}".format(roots.shape))
        createRootsFuncPlotWithLines(roots, L, omega, wLO, wTO, epsInf, mode, "Roots")
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

