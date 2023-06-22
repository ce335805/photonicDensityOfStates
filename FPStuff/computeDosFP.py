import numpy as np
import scipy.constants as consts
import math


def nzVals(omega, L):
    nMax = np.floor(L * omega / np.pi / consts.c)
    nArr = np.arange(nMax) + 1.
    return nArr


def computeDosFPAsOfFeq(zVal, omegaArr, L):

    dos = np.zeros(omegaArr.shape)
    for wInd, wVal in enumerate(omegaArr):
        dos[wInd] = computeDosFPAsOfZ(np.array([zVal]), wVal, L)
    return dos


def computeDosFPAsOfZ(zArr, omegaVal, L):
    nzArr = nzVals(omegaVal, L)
    prefac = omegaVal / (2. * np.pi * L * consts.c ** 2)
    bracket = 1. + consts.c ** 2 * np.pi ** 2 * nzArr[None, :] ** 2 / omegaVal ** 2 / L ** 2
    sinTerm = np.sin(np.pi / L * nzArr[None, :] * zArr[:, None])**2
    dos = np.sum(prefac * bracket * sinTerm, axis=1)
    return dos