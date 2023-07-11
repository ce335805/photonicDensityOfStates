import numpy as np
from scipy.integrate import quad
import math

import scipy.constants as consts

import understandThings.pedestrianFP.allowedKKPara as allowedKsK
import understandThings.pedestrianFP.plotPedestriandoss as plot

def dispersionK(kx, ky, kz):
    return consts.c * np.sqrt(kx**2 + ky**2 + kz**2)


def dosPedestrian(kxArr, kyArr, zArr, omega, deltaOmega, eps, d, L):
    dos = np.zeros(zArr.shape)
    print("Computing dos via sum")
    for kx in kxArr:
        for ky in kyArr:
            kzArr = allowedKsK.findKs(L, omega + deltaOmega, kx, ky, eps, "TE")
            for kz in kzArr:
                omegaTry = dispersionK(kx, ky, kz)
                if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                        continue
                kD = np.sqrt((eps - 1) * (kx**2 + ky**2) + eps * kz ** 2)
                NSqr = L**3 / 2 * (eps * np.sin(kz * d / 2.) ** 2 * (1 - np.sin(kD * d) / (kD * d)) + np.sin(kD * d / 2.) ** 2  * (1 - np.sin(kz * d) / (kz * d)))
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



def pedestrianFPMain():

    L = .4
    d = .1
    eps = 2.
    w1 = np.pi / d * consts.c
    w2 = 2. * np.pi / d * consts.c
    print("wMin = {}GHz".format(w1 * 1e-9))
    print("wMax = {}GHz".format(w2 * 1e-9))

    omega = 0.3 * 1e11
    delOmega = 1e-1 * omega

    zArr = np.linspace(0., d / 2., 100)
    dosPed = calcDosPedestrian(zArr, omega, delOmega, eps, d, L)
    plot.plotPedestrianDoss(zArr, dosPed, omega, delOmega, eps)

pedestrianFPMain()

