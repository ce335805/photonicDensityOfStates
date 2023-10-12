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

    #upBound = 3. * 1e14
    #omArr = np.linspace(1e12, upBound, 10000, endpoint=True)
    #L1 = 1e-4
    #freq = np.pi * consts.c / L1
    #print("base freq = {}THz".format(freq * 1e-12))
    #print("number of Points minimum: {}".format(upBound / freq))
    #dosFP1 = computeDosFP.dosParallelAsOfFeq(np.array([L1 / 2.]), omArr, L1)
    ##integratedDos.computeIntegral(omArr, dosParallel, L1)
#
    #L2 = 1e-5
    #freq = np.pi * consts.c / L2
    #print("base freq = {}THz".format(freq * 1e-12))
    #print("number of Points minimum: {}".format(upBound / freq))
    #dosFP2 = computeDosFP.dosParallelAsOfFeq(np.array([L2 / 2.]), omArr, L2)
    #plotDosFP.plotDosCompare(omArr, dosFP1, dosFP2, L1, L2)


    #compute field plots
    evCutoff = 241.8 * 1e12
    cutoff = 2. * evCutoff
    dArr = np.logspace(-6, -3, 100, endpoint=True)
    #fluctuationsE = integratedDos.numericalIntegralEField(cutoff, dArr)
    filename = "fluctuationsE"
    #handleIntegralData.writeDataFixedCutoff(cutoff, dArr, fluctuationsE, filename)
    cutoff, dArr, fluctuationsE = handleIntegralData.retrieveDataFixedCutoff(filename)
    #fluctuationsA = integratedDos.numericalIntegralAField(cutoff, dArr)
    filename = "fluctuationsA"
    #handleIntegralData.writeDataFixedCutoff(cutoff, dArr, fluctuationsA, filename)
    cutoff, dArr, fluctuationsA = handleIntegralData.retrieveDataFixedCutoff(filename)
    baseFreqArr = np.pi * consts.c / dArr

    #plotDosFP.plotFluctuationsEAsOfD(dArr, fluctuationsE, baseFreqArr, cutoff)
    #plotDosFP.plotFluctuationsAAsOfD(dArr, fluctuationsA, baseFreqArr, cutoff)
    plotDosFP.plotFluctuationsAandE(dArr, fluctuationsE, fluctuationsA, baseFreqArr, cutoff)

    #exit()

    #dArr = np.logspace(-3, -6, 100)
    #cutoff = 241.8 * 1e12
    #filename = "hoppingFixedCutoff"
    ##hopArr = integratedDos.numericalIntegralHopping(cutoff, dArr, )
    ##handleIntegralData.writeDataFixedCutoff(cutoff, dArr, hopArr, filename)
    #cutoff, dArr, hopArr = handleIntegralData.retrieveDataFixedCutoff(filename)
    #freqArr = np.pi * consts.c / dArr
    #plotDosFP.plotFluctuationsEAsOfD(dArr, hopArr, freqArr)


dosFPMain()
