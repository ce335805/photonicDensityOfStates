import numpy as np
from scipy.integrate import quad
import math

import scipy.constants as consts

import understandThings.pedestrianFP.allowedKsOm as allowedKsOm
import understandThings.pedestrianFP.allowedKKParaDiff as diff

def modeFuncTEPos(kzArr, kDArr, zArr, d):
    return np.sin(kzArr[None, :] * (d / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * d / 2.)

def modeFuncTENeg(kzArr, kDArr, zArr, d):
    return np.sin(kDArr[None, :] * (d / 2 + zArr[:, None])) * np.sin(kzArr[None, :] * d / 2.)

def normalizationNSq(kzArr, kDArr, d, eps):
    return d / 2 * (eps * np.sin(kzArr * d / 2.)**2  * (1 - np.sin(kDArr * d) / (kDArr * d)) + np.sin(kDArr * d / 2.)**2 * (1 - np.sin(kzArr * d) / (kzArr * d)))

def splitZArrPosNeg(zArr):
    posInd = np.where(zArr >= 0)
    zPosArr = zArr[posInd]
    negInd = np.where(zArr < 0)
    zNegArr = zArr[negInd]
    return (zNegArr, zPosArr)

def dosAnalytical(omega, zArr, eps, d):
    kzArr = allowedKsOm.findKs(d, omega, eps, "TE")
    diffArr = diff.findDerivativeKPara(d, omega, kzArr, eps)
    kDArr = np.sqrt((eps - 1) * omega**2 / consts.c**2 + kzArr**2)
    NSqr = normalizationNSq(kzArr, kDArr, d, eps)
    zNegArr, zPosArr = splitZArrPosNeg(zArr)
    funcNeg = modeFuncTENeg(kzArr, kDArr, zNegArr, d)
    funcPos = modeFuncTEPos(kzArr, kDArr, zPosArr, d)
    func = np.append(funcNeg, funcPos, axis = 0)
    prefac = omega / (2. * np.pi * consts.c**2)
    kParaArr = np.sqrt(omega**2 / consts.c**2 - kzArr**2)
    diffFac = 1. / (1. + kzArr / kParaArr * diffArr)
    return np.sum(prefac / NSqr[None, :] * func ** 2 * diffFac[None, :], axis=1)


def dosAnalyticalInt(zArr, omega, deltaOmega, eps, d):
    print("computing dos analytically")
    omArr = np.linspace(omega, omega + deltaOmega, 100)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        dosInt[:, wInd] = dosAnalytical(wVal, zArr, eps, d)

    return np.trapz(dosInt, omArr, axis=1)

def dosAnalyticalWDiff(omega, zArr, eps, d):
    kzArr = allowedKsOm.findKs(d, omega, eps, "TE")
    kzArrDel = allowedKsOm.findKsDerivativeW(d, omega, eps, "TE")
    kDArr = np.sqrt((eps - 1) * omega**2 / consts.c**2 + kzArr**2)
    NSqr = normalizationNSq(kzArr, kDArr, d, eps)
    zNegArr, zPosArr = splitZArrPosNeg(zArr)
    funcNeg = modeFuncTENeg(kzArr, kDArr, zNegArr, d)
    funcPos = modeFuncTEPos(kzArr, kDArr, zPosArr, d)
    func = np.append(funcNeg, funcPos, axis=0)
    prefac = omega / (2. * np.pi * consts.c**2)
    diffFac = (1. - consts.c ** 2 * kzArr / omega * kzArrDel)
    return np.sum(prefac / NSqr[None, :] * func ** 2 * diffFac[None, :], axis=1)


def dosAnalyticalIntWDiff(zArr, omega, deltaOmega, eps, d):
    print("computing dos analytically")
    omArr = np.linspace(omega, omega + deltaOmega, 3)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        dosInt[:, wInd] = dosAnalyticalWDiff(wVal, zArr, eps, d)

    return np.trapz(dosInt, omArr, axis=1)