import numpy
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
    term1 = k * np.sin(kD * L / 2)
    term2 = kD * np.cos(kD * L / 2) * np.tanh(k * L / 2)
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

def getRoots(L, omega, wLO, wTO, epsInf, mode):
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

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    NDiscrete = 1 * int(omega / consts.c * (np.sqrt(np.abs(eps)) + 1.) * L + 17)
    iteration = 0
    lenIntervalsOld = 0
    intervals = np.zeros((0, 2))
    while(iteration < 10):
        print("NDiscrete = {} at iteration {}".format(NDiscrete, iteration))
        subdivision = np.linspace(lowerBound, upperBound, NDiscrete, endpoint=True)
        rootFuncAtPoints = rootFunc(subdivision, L, omega, wLO, wTO, epsInf)
        signs = rootFuncAtPoints[:-1] * rootFuncAtPoints[1:]
        indsSigns = np.where(signs < 0)[0]
        intervals = np.append([subdivision[indsSigns]], [subdivision[indsSigns + 1]], axis = 0)
        if(lenIntervalsOld == intervals.shape[0]):
            break
        else:
            NDiscrete = 2 * NDiscrete
            lenIntervalsOld = intervals.shape[0]
            iteration += 1

    intervals = numpy.swapaxes(intervals, 0, 1)

    nRoots = intervals.shape[0]
    roots = np.zeros(nRoots)
    for rootInd, root in enumerate(roots):
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
    roots = getRoots(L, omega, wLO, wTO, epsInf, mode)
    createRootsFuncPlotWithLines(roots, L, omega, wLO, wTO, epsInf, mode, "Roots")
    print("Number of roots found for {} mode = {}".format(mode, len(roots)))
    return roots

def findKsDerivativeW(roots, L, omega, wLO, wTO, epsInf, mode):
    delOm = omega * 1e-5
    rootsPlus = getRoots(L, omega + delOm, wLO, wTO, epsInf, mode)
    rootsPlus = rootsPlus[:len(roots)]
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

