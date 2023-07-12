import numpy as np
import scipy.constants as consts
import scipy.optimize

def rootFuncTE(k, kparaSq, d, eps):
    kD = np.sqrt((eps - 1) * kparaSq + eps * k**2)
    term1 = k * np.cos(k * d / 2.) * np.sin(kD * d / 2.)
    term2 = kD * np.cos(kD * d / 2.) * np.sin(k * d / 2.)
    return  term1 + term2

def computeSingleDerivative(d, kPara, delK, kz, eps):
    delKz = 5. * delK
    rootPlus = scipy.optimize.root_scalar(rootFuncTE, args=((kPara + delK)**2, d, eps), bracket=(kz - delKz, kz + delKz)).root
    rootMinus = scipy.optimize.root_scalar(rootFuncTE, args=((kPara - delK)**2, d, eps), bracket=(kz - delKz, kz + delKz)).root
    return (rootPlus - rootMinus) / (2. * delK)

def findDerivativeKPara(d, omega, kzArr, eps):
    kParaArr = np.sqrt(omega**2 / consts.c**2 - kzArr**2)
    delK = kzArr[0] * 1e-3
    diffArr = np.zeros(kzArr.shape)
    for pairInd in range(len(kzArr)):
        diffArr[pairInd] = computeSingleDerivative(d, kParaArr[pairInd], delK, kzArr[pairInd], eps)
    return diffArr

