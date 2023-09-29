import numpy as np
import scipy.constants as consts

import computeDosFP
import plotDosFP
import effectiveMassFP
import integratedDos
import handleIntegralData

def dosForSeveralOmAsOfZ(zArr, omegaArr, L):
    dosOms = np.zeros((len(zArr), len(omegaArr)))
    for omInd, omVal in enumerate(omegaArr):
        dosOms[:, omInd] = computeDosFP.computeDosFPAsOfZ(zArr, omVal, L)
    return dosOms


def dosFPMain():

    print("Computing stuff for the FP")

#    omegaArr = np.array([1., 5., 10., 20.]) * 1e13
#    zArr = np.linspace(0., L, 10000, endpoint = True)
#    dosFPForOmegas = dosForSeveralOmAsOfZ(zArr, omegaArr, L)
#    plotDosFP.plotDosAsOfZSeveralOm(dosFPForOmegas, zArr, omegaArr, L)
#
    #upBound = 3. * 1e14
    #omArr = np.linspace(1e12, upBound, 10000, endpoint=True)
    #L1 = 1e-4
    #freq = np.pi * consts.c / L1
    #print("base freq = {}THz".format(freq * 1e-12))
    #print("number of Points minimum: {}".format(upBound / freq))

    #dosFP1 = computeDosFP.dosTEAsOfFeq(np.array([L1 / 2.]), omArr, L1)
    #cutoffArr, FieldInt = integratedDos.numericalIntegral(L1 / 2, upBound, L1)
    #handleIntegralData.writeData(cutoffArr, FieldInt, "dosTME")
    #cutoffArr, FieldInt = handleIntegralData.retrieveData("dosTME")
    #plotDosFP.plotFieldIntegrals(cutoffArr, FieldInt)


    #dosParallel = computeDosFP.dosParallelAsOfFeq(np.array([L1 / 2.]), omArr, L1)
    #integratedDos.computeIntegral(omArr, dosParallel, L1)

    #L2 = 1e-5
    #dosFP2 = computeDosFP.dosTEAsOfFeq(np.array([L2 / 2.]), omArr, L2)
    #plotDosFP.plotDosCompare(omArr, dosFP1, dosFP2, L1, L2)


#    Lambda = 100 * 1e12
#    omArr = np.logspace(11., np.log10(Lambda), 10000, base=10., endpoint=True)
#    #omArr = np.linspace(1e10, 1e15, 10000, endpoint=True)
#    dtEff = effectiveMassFP.integrateDosDifference(L, omArr)
#    print("dt = {}".format(dtEff))
#    effectiveMassFP.plotEffectiveMassIntegrad(L, omArr)

#### compute field strength using a fixed cutoff

    #dArr = np.logspace(-3, -6, 100)
    #cutoff = 241.8 * 1e12
    #filename = "dataFixedCutoff"
    ##fieldArr = integratedDos.numericalIntegralFixedCutoff(cutoff, dArr, )
    ##fieldArr = integratedDos.numericalIntegralFixedCutoff(cutoff, dArr, )
    ##handleIntegralData.writeDataFixedCutoff(cutoff, dArr, fieldArr, filename)
    #cutoff, dArr, fieldArr = handleIntegralData.retrieveDataFixedCutoff(filename)
    #freqArr = np.pi * consts.c / dArr
    #plotDosFP.plotFieldWithFixedCutoff(dArr, fieldArr, freqArr)


    #would like some estimate for diamagnetic shift
    lz = 1e-6
    print("me = {}".format(consts.m_e))
    shift = 4. * np.pi * consts.fine_structure * consts.hbar * consts.c / consts.m_e * 1e20 * 1. / lz
    print("diamagnetic shift for lz = {}m: om = {}THz".format(lz, np.sqrt(shift) * 1e-12))
    exit()
#### compute effective hopping using a fixed cutoff

    dArr = np.logspace(-3, -6, 100)
    cutoff = 241.8 * 1e12
    filename = "hoppingFixedCutoff"
    #hopArr = integratedDos.numericalIntegralHopping(cutoff, dArr, )
    #handleIntegralData.writeDataFixedCutoff(cutoff, dArr, hopArr, filename)
    cutoff, dArr, hopArr = handleIntegralData.retrieveDataFixedCutoff(filename)
    freqArr = np.pi * consts.c / dArr
    plotDosFP.plotHoppingWithFixedCutoff(dArr, hopArr, freqArr)


dosFPMain()
