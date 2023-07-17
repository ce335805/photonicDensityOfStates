import numpy as np
from scipy.integrate import quad
import math

import scipy.constants as consts

import understandThings.pedestrianFP.allowedKsOm as allowedKsOm
import understandThings.pedestrianFP.allowedKKParaDiff as diff

def modeFuncTEPos(kzArr, kDArr, zArr, d):
    return np.sinh(kzArr[None, :] * (d / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * d / 2.)

def modeFuncTENeg(kzArr, kDArr, zArr, d):
    return np.sin(kDArr[None, :] * (d / 2 + zArr[:, None])) * np.sinh(kzArr[None, :] * d / 2.)

def normalizationNSq(kzArr, kDArr, d, eps):
    return d / 2 * (eps * np.sinh(kzArr * d / 2.)**2  * (1 - np.sin(kDArr * d) / (kDArr * d)) + np.sin(kDArr * d / 2.)**2 * (np.sinh(kzArr * d) / (kzArr * d) - 1))

def splitZArrPosNeg(zArr):
    posInd = np.where(zArr >= 0)
    zPosArr = zArr[posInd]
    negInd = np.where(zArr < 0)
    zNegArr = zArr[negInd]
    return (zNegArr, zPosArr)

def dosAnalyticalWDiff(omega, zArr, eps, d):
    kzArr = allowedKsOm.findKs(d, omega, eps, "TEEva")
    kzArrDel = allowedKsOm.findKsDerivativeW(d, omega, eps, "TEEva")
    kDArr = np.sqrt((eps - 1) * omega**2 / consts.c**2 - kzArr**2)
    NSqr = normalizationNSq(kzArr, kDArr, d, eps)
    zNegArr, zPosArr = splitZArrPosNeg(zArr)
    funcNeg = modeFuncTENeg(kzArr, kDArr, zNegArr, d)
    funcPos = modeFuncTEPos(kzArr, kDArr, zPosArr, d)
    func = np.append(funcNeg, funcPos, axis=0)
    prefac = omega / (2. * np.pi * consts.c**2)
    diffFac = (1. + consts.c ** 2 * kzArr / omega * kzArrDel)
    print(diffFac)
    return np.sum(prefac / NSqr[None, :] * func ** 2 * diffFac[None, :], axis=1)


def dosAnalyticalIntWDiff(zArr, omega, deltaOmega, eps, d):
    print("computing dos analytically")
    omArr = np.linspace(omega, omega + deltaOmega, 3)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        dosInt[:, wInd] = dosAnalyticalWDiff(wVal, zArr, eps, d)

    return np.trapz(dosInt, omArr, axis=1)

def dosAnalyticalkD(omega, zArr, eps, d):
    kDArr = allowedKsOm.findKs(d, omega, eps, "TEEvakD")
    kDArrDel = allowedKsOm.findKsDerivativeW(d, omega, eps, "TEEvakD")
    kzArr = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kDArr ** 2)
    NSqr = normalizationNSq(kzArr, kDArr, d, eps)
    zNegArr, zPosArr = splitZArrPosNeg(zArr)
    funcNeg = modeFuncTENeg(kzArr, kDArr, zNegArr, d)
    funcPos = modeFuncTEPos(kzArr, kDArr, zPosArr, d)
    func = np.append(funcNeg, funcPos, axis=0)
    prefac = eps * omega / (2. * np.pi * consts.c ** 2)
    diffFac = (1. - consts.c ** 2 * kDArr / (eps * omega) * kDArrDel)
    print(diffFac)
    return np.sum(prefac / NSqr[None, :] * func ** 2 * diffFac[None, :], axis=1)

def dosAnalyticalIntkD(zArr, omega, deltaOmega, eps, d):
    print("computing dos analytically")
    omArr = np.linspace(omega, omega + deltaOmega, 3)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        dosInt[:, wInd] = dosAnalyticalkD(wVal, zArr, eps, d)


    return np.trapz(dosInt, omArr, axis=1)