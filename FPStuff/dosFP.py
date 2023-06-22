import numpy as np

import computeDosFP
import plotDosFP
import effectiveMassFP

def dosFPMain():

    omega = 1 * 1e10
    L = 1.

    #zArr = np.linspace(0., L, 10000, endpoint = True)
    #dosFPAsOfZ = computeDosFP.computeDosFPAsOfZ(zArr, omega, L)
    #plotDosFP.plotDosAsOfZ(zArr, dosFPAsOfZ, omega, L)

    omArr = np.logspace(8., 11., 5000, base = 10., endpoint=True)
    dosFPAsOfOm = computeDosFP.computeDosFPAsOfFeq(L / 2., omArr, L)
    plotDosFP.plotDosAsOfOm(omArr, dosFPAsOfOm, L)

    Lambda = 1e10
    omArr = np.logspace(5., np.log10(Lambda), 50000, base=10., endpoint=True)
    dtEff = effectiveMassFP.integrateDosDifference(L, omArr)
    print("dt = {}".format(dtEff))

    effectiveMassFP.plotEffectiveMassIntegrad(L, omArr)

dosFPMain()
