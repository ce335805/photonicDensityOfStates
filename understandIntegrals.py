import numpy as np
import scipy.constants as consts
import scipy.integrate as integrate

import findAllowedKs

def waveFunctionPos(zArr, kVal, L, omega, eps):
    kDVal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)
    func = np.sin(kVal * (L / 2 - zArr)) * np.sin(kDVal * L / 2.)
    return func

def waveFunctionNeg(zArr, kVal, L, omega, eps):
    kDVal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)
    func = np.sin(kDVal * (L / 2. + zArr)) * np.sin(kVal * L / 2.)
    return func

def productWavefunctionsPos(zArr, k1Val, k2Val, L, omega1, omega2, eps):
    return waveFunctionPos(zArr, k1Val, L, omega1, eps) * waveFunctionPos(zArr, k2Val, L, omega2, eps)

def productWavefunctionsNeg(zArr, k1Val, k2Val, L, omega1, omega2, eps):
    return waveFunctionNeg(zArr, k1Val, L, omega1, eps) * waveFunctionNeg(zArr, k2Val, L, omega2, eps)

def integralPos(k1Val, k2Val, L, omega1, omega2, eps):
    return integrate.quad(productWavefunctionsPos, 0, L / 2., args = (k1Val, k2Val, L, omega1, omega2, eps))

def integralNeg(k1Val, k2Val, L, omega1, omega2, eps):
    return integrate.quad(productWavefunctionsNeg, -L / 2., 0, args = (k1Val, k2Val, L, omega1, omega2, eps))

def integralPosAna(k1Val, k2Val, L, omega1, omega2, eps):
    k1DVal = np.sqrt((eps - 1) * omega1 ** 2 / consts.c ** 2 + k1Val ** 2)
    k2DVal = np.sqrt((eps - 1) * omega2 ** 2 / consts.c ** 2 + k2Val ** 2)
    
    prefac = np.sin(k1DVal * L / 2.) * np.sin(k2DVal * L / 2.)
    num = k2Val * np.sin(k1Val * L / 2.) * np.cos(k2Val * L / 2.) - k1Val * np.sin(k2Val * L / 2.) * np.cos(k1Val * L / 2.)
    denom = k1Val**2 - k2Val**2

    return prefac * num / denom


def integralNegAna(k1Val, k2Val, L, omega1, omega2, eps):
    k1DVal = np.sqrt((eps - 1) * omega1 ** 2 / consts.c ** 2 + k1Val ** 2)
    k2DVal = np.sqrt((eps - 1) * omega2 ** 2 / consts.c ** 2 + k2Val ** 2)

    prefac = np.sin(k1Val * L / 2.) * np.sin(k2Val * L / 2.)
    num = k2DVal * np.sin(k1DVal * L / 2.) * np.cos(k2DVal * L / 2.) - k1DVal * np.sin(k2DVal * L / 2.) * np.cos(
        k1DVal * L / 2.)
    denom = k1DVal ** 2 - k2DVal ** 2

    return prefac * num / denom

def findKsTE(L, omega, eps):
    NDiscrete = 10 * int(omega / consts.c * L / (4. * np.pi) + 17)
    extremaTE = findAllowedKs.extremalPoints(L, omega, eps, NDiscrete, "TE")
    rootsTE = findAllowedKs.computeRootsGivenExtrema(L, omega, eps, extremaTE, "TE")
    print("Number of roots for TEE found = {}".format(rootsTE.shape))
    return rootsTE


def understandMain():

    print("Lets understand some integrals")
    epsilon = 2.
    omega = 1 * 1e11
    L = 0.1

    allowedKs = findKsTE(L, omega, epsilon)

    k1 = allowedKs[1]
    k2 = allowedKs[2]

    kpara = 0.

    omega1 = np.sqrt(consts.c ** 2 * (k1 ** 2 + kpara ** 2))
    omega2 = np.sqrt(consts.c ** 2 * (k2 ** 2 + kpara ** 2))

    intPos = integralPos(allowedKs[1], allowedKs[2], L, omega1, omega2, epsilon)
    intNeg = integralNeg(allowedKs[1], allowedKs[2], L, omega1, omega2, epsilon)

    print("intPos = {}".format(intPos))
    print("intNeg = {}".format(intNeg))

    print("intPos + intNeg = {}".format(intPos[0] + intNeg[0]))
    print("intPos + eps * intNeg = {}".format(intPos[0] + epsilon * intNeg[0]))

    intPosAna = integralPosAna(allowedKs[1], allowedKs[2], L, omega1, omega2, epsilon)
    intNegAna = integralNegAna(allowedKs[1], allowedKs[2], L, omega1, omega2, epsilon)

    print("intPosAna = {}".format(intPosAna))
    print("intNegAna = {}".format(intNegAna))

understandMain()