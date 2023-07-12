import numpy as np
from scipy.integrate import quad
import math

import scipy.constants as consts

import understandThings.pedestrianFP.allowedKKPara as allowedKsK
import understandThings.pedestrianFP.allowedKsOm as allowedKsOm
import understandThings.pedestrianFP.plotPedestriandoss as plot
import understandThings.pedestrianFP.allowedKKParaDiff as diff


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
    kParaArr = np.linspace(0, (omega + deltaOmega) / consts.c, 200)
    dos = np.zeros((len(zArr), len(kParaArr)))
    plotAlready = False
    for kParaInd, kParaVal in enumerate(kParaArr):
        kzArr = allowedKsK.findKsKPara(d, omega + deltaOmega, kParaVal**2, eps, "TE", False)
        for kz in kzArr:
            omegaTry = dispersionKKPara(kParaVal, kz)
            if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                    continue
            if(not plotAlready):
                kzArr = allowedKsK.findKsKPara(d, omega + deltaOmega, kParaVal ** 2, eps, "TE", True)
                plotAlready = True
            kD = np.sqrt((eps - 1) * kParaVal**2 + eps * kz ** 2)
            NSqr = d / 2 * (eps * np.sin(kz * d / 2.) ** 2 * (1 - np.sin(kD * d) / (kD * d)) + np.sin(kD * d / 2.) ** 2  * (1 - np.sin(kz * d) / (kz * d)))
            dos[:, kParaInd] += 1 / NSqr * np.sin(kz * (d / 2. - zArr))**2 * np.sin(kD * d / 2.) ** 2
    dosSummed = np.trapz(1. / (2. * np.pi) * kParaArr[None, :] * dos[:, :], kParaArr, axis = 1)
    return dosSummed

def dosAnalytical(omega, zArr, eps, d, plotRoots):
    kzArr = allowedKsOm.findKs(d, omega, eps, "TE", plotRoots)
    diffArr = diff.findDerivativeKPara(d, omega, kzArr, eps)
    #print("diffArr = {}".format(diffArr))
    #kzArrDel = allowedKsOm.findKsDerivativeW(d, omega, eps, "TE")
    kDArr = np.sqrt((eps - 1) * omega**2 / consts.c**2 + kzArr**2)
    NSqr = d / 2 * (eps * np.sin(kzArr * d / 2.)**2  * (1 - np.sin(kDArr * d) / (kDArr * d)) + np.sin(kDArr * d / 2.)**2 * (1 - np.sin(kzArr * d) / (kzArr * d)))
    func = np.sin(kzArr[None, :] * (d / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * d / 2.)
    prefac = omega / (2. * np.pi * consts.c**2)
    kParaArr = np.sqrt(omega**2 / consts.c**2 - kzArr**2)
    diffFac = kParaArr / (kParaArr + kzArr * diffArr)
    return np.sum(prefac / NSqr[None, :] * func ** 2 * diffFac[None, :], axis=1)


def dosAnalyticalInt(zArr, omega, deltaOmega, eps, d):
    print("computing dos analytically")
    omArr = np.linspace(omega, omega + deltaOmega, 20)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        plotRoots = (wInd == 0)
        dosInt[:, wInd] = dosAnalytical(wVal, zArr, eps, d, plotRoots)

    return np.trapz(dosInt, omArr, axis=1)


def pedestrianFPMain():

    L = .8
    d = .2
    eps = 2.
    w1 = np.pi / d * consts.c
    w2 = 2. * np.pi / d * consts.c
    print("wMin = {}GHz".format(w1 * 1e-9))
    print("wMax = {}GHz".format(w2 * 1e-9))

    omega = 1. * 1e11
    delOmega = 1. * 1e-2 * omega

    zArr = np.linspace(0., d / 2., 100)

    dosAna = dosAnalyticalInt(zArr, omega, delOmega, eps, d)
    #dosPed = calcDosPedestrian(zArr, omega, delOmega, eps, d, L)
    dosPedInt = dosPedestrianInt(zArr, omega, delOmega, eps, d)
    plot.plotPedestrianDoss(zArr, dosPedInt, dosPedInt, dosAna, omega, delOmega, eps)
    #plot.plotPedestrianDoss(zArr, dosAna, dosAna, dosAna, omega, delOmega, eps)

pedestrianFPMain()

