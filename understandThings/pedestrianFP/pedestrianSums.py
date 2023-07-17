import numpy as np
from scipy.integrate import quad
import math

import scipy.constants as consts

import understandThings.pedestrianFP.allowedKKPara as allowedKsK

def dispersionK(kx, ky, kz):
    return consts.c * np.sqrt(kx**2 + ky**2 + kz**2)

def dispersionKKPara(kPara, kz):
    return consts.c * np.sqrt(kPara**2 + kz**2)

def dosPedestrian(kxArr, kyArr, zArr, omega, deltaOmega, eps, d, L):
    dos = np.zeros(zArr.shape)
    print("Computing dos via sum")
    for kx in kxArr:
        for ky in kyArr:
            kzArr = allowedKsK.findKs(d, omega + deltaOmega, kx, ky, eps, "TE")
            for kz in kzArr:
                omegaTry = dispersionK(kx, ky, kz)
                if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                        continue
                kD = np.sqrt((eps - 1) * (kx**2 + ky**2) + eps * kz ** 2)
                NSqr = L**2 * d / 2 * (eps * np.sin(kz * d / 2.) ** 2 * (1 - np.sin(kD * d) / (kD * d)) + np.sin(kD * d / 2.) ** 2  * (1 - np.sin(kz * d) / (kz * d)))
                dos += 1 / NSqr * np.sin(kz * (d / 2. - zArr))**2 * np.sin(kD * d / 2.) ** 2
    return dos

def calcDosPedestrian(zArr, omega, deltaOmega, eps, d, L):

    kBound = np.sqrt(eps) * (omega + deltaOmega) / consts.c
    nBound = math.ceil(kBound * L / (2. * np.pi))
    kxInds = np.arange(- nBound, nBound)
    kxArr = kxInds * 2. * np.pi / L
    kyArr = kxArr
    dos = dosPedestrian(kxArr, kyArr, zArr, omega, deltaOmega, eps, d, L)
    return dos


def dosPedestrianInt(zArr, omega, deltaOmega, eps, d):
    print("Computing dos via kPara Integral")
    kParaArr = np.linspace(0, (omega + deltaOmega) / consts.c, 1000)
    dos = np.zeros((len(zArr), len(kParaArr)))
    for kParaInd, kParaVal in enumerate(kParaArr):
        kzArr = allowedKsK.findKsKPara(d, omega + deltaOmega, kParaVal**2, eps, "TE")
        for kz in kzArr:
            omegaTry = dispersionKKPara(kParaVal, kz)
            if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                    continue
            kD = np.sqrt((eps - 1) * kParaVal**2 + eps * kz ** 2)
            NSqr = d / 2 * (eps * np.sin(kz * d / 2.) ** 2 * (1 - np.sin(kD * d) / (kD * d)) + np.sin(kD * d / 2.) ** 2  * (1 - np.sin(kz * d) / (kz * d)))
            dos[:, kParaInd] += 1 / NSqr * np.sin(kz * (d / 2. - zArr))**2 * np.sin(kD * d / 2.) ** 2
    dosSummed = np.trapz(1. / (2. * np.pi) * kParaArr[None, :] * dos[:, :], kParaArr, axis = 1)
    return dosSummed