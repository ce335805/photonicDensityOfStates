import numpy as np
import scipy.constants as consts
import math

def modeFuncSqTEPara(kzArr, zVal, d):
    func = np.sin(kzArr * zVal)**2
    norm = 2 / d / (1 - np.sin(2. * kzArr * d) / (2. * kzArr * d))
    return func * norm

def modeFuncSqTMPara(kzArr, qArr, zVal, d):
    func = np.sin(kzArr * zVal)**2
    norm = 2. / d / (( 1 + qArr[None, :]**2 / kzArr[:, None]**2) + ( qArr[None, :]**2 / kzArr[:, None]**2 - 1) * np.sin(2. * kzArr[:, None] * d) / (2. * kzArr[:, None] * d))

    return func[:, None] * norm[:, :]

def getKzArr(d):
    nmax = math.ceil(100. * 240 * 1e12 / consts.c * d / np.pi)
    nArr = np.arange(1, nmax)
    return nArr * np.pi / d

def computeAlpha2FTransLong(d, zVal, qArr, OmArr):

    kzArr = getKzArr(d)
    print("Number of kz values at d = {}m: {}".format(d, len(kzArr)))
    omegaArr = consts.c * np.sqrt(kzArr[:, None]**2 + qArr[None, :]**2)

    cutoffArr = (omegaArr > 5. * 240 * 1e12).astype(float)

    modeTE = modeFuncSqTEPara(kzArr, zVal, d)
    modeTM = modeFuncSqTMPara(kzArr, qArr, zVal, d)

    alpha2FTrans = np.sum(modeTE[:, None, None] / (OmArr[None, None, :]**2 + omegaArr[:, :, None]**2), axis = 0)
    alpha2FLong = np.sum(modeTM[:, :, None] / (OmArr[None, None, :] ** 2 + omegaArr[:, :, None] ** 2), axis = 0)

    #alpha2FTrans = np.sum(1. / (OmArr[None, None, :] ** 2 + omegaArr[:, :, None] ** 2) / d, axis = 0)
    #alpha2FLong = np.sum(1. / (OmArr[None, None, :] ** 2 + omegaArr[:, :, None] ** 2) / d, axis = 0)

    prefac = 2. * np.pi * consts.fine_structure * consts.hbar**3 * consts.c / consts.m_e ** 2 / consts.e

    prefac = prefac * 2. * np.pi#for fermi wave-vector

    return (prefac * alpha2FTrans, prefac * alpha2FLong)

def computeAlpha2FTransLongD(dArr, qArr, OmArr):
    alpha2FTrans = np.zeros((len(dArr), len(qArr), len(OmArr)))
    alpha2FLong = np.zeros((len(dArr), len(qArr), len(OmArr)))

    for dInd, dVal in enumerate(dArr):
        alpha2FTrans[dInd, :, :], alpha2FLong[dInd, :, :] = computeAlpha2FTransLong(dVal, dVal / 2., qArr, OmArr)

    return(alpha2FTrans, alpha2FLong)

