import numpy as np
import scipy.constants as consts
import math


def nzValsTE(omega, L):
    nMax = np.floor(L * omega / np.pi / consts.c)
    nArr = np.arange(nMax) + 1
    return nArr

def nzValsTM(omega, L):
    nMax = np.floor(L * omega / np.pi / consts.c)
    nArr = np.arange(nMax + 1)
    return nArr

def computeDosFPAsOfFeq(zVal, omegaArr, L):

    dos = np.zeros(omegaArr.shape)
    for wInd, wVal in enumerate(omegaArr):
        dos[wInd] = computeDosFPAsOfZ(np.array([zVal]), wVal, L)
    return dos


def computeDosFPAsOfZ(zArr, omegaVal, L):
    nzArr = nzValsTM(omegaVal, L)
    prefac = omegaVal / (2. * np.pi * L * consts.c ** 2)
    bracket = 1. + consts.c ** 2 * np.pi ** 2 * nzArr[None, :] ** 2 / omegaVal ** 2 / L ** 2
    sinTerm = np.sin(np.pi / L * nzArr[None, :] * zArr[:, None])**2
    dos = np.sum(prefac * bracket * sinTerm, axis=1)
    return dos

def dosTEAsOfFeq(zArr, omegaArr, L):
    dos = np.zeros(omegaArr.shape)
    for wInd, wVal in enumerate(omegaArr):
        dos[wInd] = dosTE(zArr, wVal, L)
    return dos

def dosTE(zArr, omega, L):
    nzArr = nzValsTE(omega, L)
    kzVals = np.pi / L * nzArr
    normSqr = L / 2. * (1. - np.sin(2. * kzVals * L) / (2. * kzVals * L))
    prefac = np.pi * consts.c / (2. * omega)
    dos = prefac * 1 / normSqr[None, :] * np.sin(kzVals[None, :] * zArr[:, None])**2
    return np.sum(dos, axis = 1)

def dosTMAsOfFeq(zArr, omegaArr, L):
    dos = np.zeros(omegaArr.shape)
    for wInd, wVal in enumerate(omegaArr):
        dos[wInd] = dosTM(zArr, wVal, L)
    return dos

def dosTM(zArr, omega, L):
    nzArr = nzValsTM(omega, L)
    kzVals = np.pi / L * nzArr
    normSqr = np.zeros(len(kzVals))
    normSqr[0] = L  * omega**2
    normSqr[1:] = L / 2. * (omega**2 + (omega**2 - 2 * consts.c**2 * kzVals[1:]**2) * np.sin(2. * kzVals[1:] * L) / (2. * kzVals[1:] * L))
    prefac = np.pi * consts.c / (2. * omega)
    dos = prefac * 1 / normSqr[None, :] * ( consts.c**2 * kzVals**2 * np.sin(kzVals[None, :] * zArr[:, None])**2 + (omega**2 - consts.c**2 * kzVals**2) * np.cos(kzVals[None, :] * zArr[:, None])**2)
    return np.sum(dos[:, 0:], axis = 1)

def dosTMParallel(zArr, omega, L):
    nzArr = nzValsTM(omega, L)
    kzVals = np.pi / L * nzArr
    if(len(kzVals) == 0):
        return 0.
    normSqr = np.zeros(len(kzVals))
    normSqr[0] = L / 2. * (omega**2 + (omega**2 - 2 * consts.c**2 * kzVals[0]**2))
    normSqr[1:] = L / 2. * (omega**2 + (omega**2 - 2 * consts.c**2 * kzVals[1:]**2) * np.sin(2. * kzVals[1:] * L) / (2. * kzVals[1:] * L))
    prefac = np.pi * consts.c / (2. * omega)
    dos = prefac * 1 / normSqr[None, :] * ( consts.c**2 * kzVals**2 * np.sin(kzVals[None, :] * zArr[:, None])**2 )
    return np.sum(dos, axis = 1)

def dosTMPerp(zArr, omega, L):
    nzArr = nzValsTM(omega, L)
    kzVals = np.pi / L * nzArr
    if (len(kzVals) == 0):
        return 0.
    normSqr = np.zeros(len(kzVals))
    normSqr[0] = L / 2. * (omega ** 2 + (omega ** 2 - 2 * consts.c ** 2 * kzVals[0] ** 2))
    normSqr[1:] = L / 2. * (
                omega ** 2 + (omega ** 2 - 2 * consts.c ** 2 * kzVals[1:] ** 2) * np.sin(2. * kzVals[1:] * L) / (
                    2. * kzVals[1:] * L))
    prefac = np.pi * consts.c / (2. * omega)
    dos = prefac * 1 / normSqr[None, :] * ((omega ** 2 - consts.c ** 2 * kzVals ** 2) * np.cos(kzVals[None, :] * zArr[:, None]) ** 2)
    return np.sum(dos, axis=1)

def dosParallelAsOfFeq(zArr, omegaArr, L):
    dos = np.zeros(omegaArr.shape)
    for wInd, wVal in enumerate(omegaArr):
        dos[wInd] = dosTE(zArr, wVal, L)
        dos[wInd] += dosTMParallel(zArr, wVal, L)
    return dos

def dosParallelOneFreq(omega, zVal, L):
        return dosTE(np.array([zVal]), omega, L) + dosTMParallel(np.array([zVal]), omega, L)

def dosPerpOneFreq(omega, zVal, L):
    return dosTMPerp(np.array([zVal]), omega, L)

def dosTEOneFreq(omega, zVal, L):
    return dosTE(np.array([zVal]), omega, L)

def dosTMOneFreq(omega, zVal, L):
    return dosTM(np.array([zVal]), omega, L)

def dosPerpAsOfFeq(zArr, omegaArr, L):
    dos = np.zeros(omegaArr.shape)
    for wInd, wVal in enumerate(omegaArr):
        dos[wInd] += dosTMPerp(zArr, wVal, L)
    return dos