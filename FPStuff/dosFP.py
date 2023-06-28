import numpy as np

import computeDosFP
import plotDosFP
import effectiveMassFP

def dosForSeveralOmAsOfZ(zArr, omegaArr, L):
    dosOms = np.zeros((len(zArr), len(omegaArr)))
    for omInd, omVal in enumerate(omegaArr):
        dosOms[:, omInd] = computeDosFP.computeDosFPAsOfZ(zArr, omVal, L)
    return dosOms


def dosFPMain():

    omega = 1 * 1e14
    L = 1e-4

#    omegaArr = np.array([1., 5., 10., 20.]) * 1e13
#    zArr = np.linspace(0., L, 10000, endpoint = True)
#    dosFPForOmegas = dosForSeveralOmAsOfZ(zArr, omegaArr, L)
#    plotDosFP.plotDosAsOfZSeveralOm(dosFPForOmegas, zArr, omegaArr, L)
#
#    omArr = np.logspace(12., 15., 5000, base = 10., endpoint=True)
#    dosFPAsOfOm = computeDosFP.computeDosFPAsOfFeq(L / 2., omArr, L)
#    plotDosFP.plotDosAsOfOm(omArr, dosFPAsOfOm, L)


    Lambda = 1e15
    omArr = np.logspace(10., np.log10(Lambda), 10000, base=10., endpoint=True)
    #omArr = np.linspace(1e10, 1e15, 10000, endpoint=True)
    dtEff = effectiveMassFP.integrateDosDifference(L, omArr)
    print("dt = {}".format(dtEff))

    effectiveMassFP.plotEffectiveMassIntegrad(L, omArr)

dosFPMain()
