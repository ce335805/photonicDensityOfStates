import numpy as np
from scipy.integrate import quad
import math

import scipy.constants as consts

import understandThings.pedestrianFP.allowedKsOm as allowedKsOm
import understandThings.pedestrianFP.plotPedestriandoss as plot
import understandThings.pedestrianFP.allowedKKParaDiff as diff

import understandThings.pedestrianFP.pedestrianSums as sums
import understandThings.pedestrianFP.omegaIntegrals as ints
import understandThings.pedestrianFP.omegaIntegralsEva as intsEva


def pedestrianFPMain():

    L = .1
    d = .1
    eps = 2.
    w1 = np.pi / d * consts.c
    w2 = 2. * np.pi / d * consts.c
    print("wMin = {}GHz".format(w1 * 1e-9))
    print("wMax = {}GHz".format(w2 * 1e-9))

    omega = 2. * 1e11
    delOmega = 1. * 1e-2 * omega

    #zArr = np.linspace(- d / 2., d / 2., 400)
    zArr = np.linspace(0., d / 2., 400)

    #dosAna = ints.dosAnalyticalInt(zArr, omega, delOmega, eps, d)
    dosAnaW = ints.dosAnalyticalIntWDiff(zArr, omega, delOmega, eps, d)
    #dosAnaEva = intsEva.dosAnalyticalIntWDiff(zArr, omega, delOmega, eps, d)
    #dosAnaEvakD = intsEva.dosAnalyticalIntkD(zArr, omega, delOmega, eps, d)
    #dosPed = sums.calcDosPedestrian(zArr, omega, delOmega, eps, d, L)
    dosPedInt = sums.dosPedestrianInt(zArr, omega, delOmega, eps, d)

    #plot.plotPedestrianDoss(zArr, dosPed, dosPed2, dosPedInt, dosPedInt2, omega, delOmega, eps)
    plot.plotPedestrianDoss(zArr, dosAnaW, dosPedInt, omega, delOmega, eps)
    #plot.plotFullDos(zArr, dosAnaW, dosAnaEva, dosAnaEvakD, omega, delOmega, eps)

pedestrianFPMain()

