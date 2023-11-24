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


    ### Compute rho as a function of omega

    #upBound = 3. * 1e14
    #omArr = np.linspace(1e12, upBound, 10000, endpoint=True)
    #L1 = 1e-4
    #freq = np.pi * consts.c / L1
    #print("base freq = {}THz".format(freq * 1e-12))
    #print("number of Points minimum: {}".format(upBound / freq))
    #dosFP1 = computeDosFP.dosParallelAsOfFeq(np.array([L1 / 2.]), omArr, L1)
    ##dosFP1 = computeDosFP.dosTEAsOfFeq(np.array([L1 / 2.]), omArr, L1) + computeDosFP.dosTMAsOfFeq(np.array([L1 / 2.]), omArr, L1)
    ##integratedDos.computeIntegral(omArr, dosParallel, L1)
    #plotDosFP.plotRhoParaAsOfOmDimensionless(omArr, dosFP1, L1)


#    L2 = 1e-5
#    freq = np.pi * consts.c / L2
#    print("base freq = {}THz".format(freq * 1e-12))
#    print("number of Points minimum: {}".format(upBound / freq))
#    dosFP2 = computeDosFP.dosParallelAsOfFeq(np.array([L2 / 2.]), omArr, L2)
#    #dosFP2 = computeDosFP.dosTEAsOfFeq(np.array([L2 / 2.]), omArr, L2) + computeDosFP.dosTMAsOfFeq(np.array([L2 / 2.]), omArr, L2)
#    #plotDosFP.plotDosCompare(omArr, dosFP1, dosFP2, L1, L2)
#    #plotDosFP.plotRhoParaAsOfOmThesis(omArr, dosFP1, dosFP2, L1, L2)

    #exit()

    ### Compute rho as function of z

    #upBound = 3. * 1e14
    #L = 1e-4
    #freq = np.pi * consts.c / L
    #omArr = np.array([4. * freq, 7. * freq])
    #zArr = np.linspace(0., L, 1000)
    #dosFP = computeDosFP.dosParallelAsOfFeqAndZ(zArr, omArr, L)
    #plotDosFP.plotRhoParaAsOfZThesis(omArr, zArr, dosFP, L)
#
    #exit()

    #evCutoff = 1519.3 * 1e12 #1eV
    #cutoffArr = np.logspace(np.log10(0.01 * evCutoff), np.log10(.5 * evCutoff), 5, endpoint=True)
    #dArr = np.logspace(-6, -4, 20, endpoint=True)
    #sumRules = np.zeros((len(cutoffArr), len(dArr)), dtype = float)
    #for cutoffInd, cutoff in enumerate(cutoffArr):
    #    sumRules[cutoffInd, :] = integratedDos.numericalIntegralSumRule(cutoff, dArr)
    ##print(sumRuleRes)
    #plotDosFP.plotSumRules(dArr, sumRules, cutoffArr)
    #exit()

    ##compute field plots
    #evCutoff = 1519.3 * 1e12 #1eV
    #cutoff = 4. * evCutoff
    #dArr = np.logspace(-6, -4, 20, endpoint=True)
    #fluctuationsE = integratedDos.numericalIntegralEField(cutoff, dArr)
    #filename = "fluctuationsE"
    #handleIntegralData.writeDataFixedCutoff(cutoff, dArr, fluctuationsE, filename)
    #cutoff, dArr, fluctuationsE = handleIntegralData.retrieveDataFixedCutoff(filename)
    #fluctuationsA = integratedDos.numericalIntegralAField(cutoff, dArr)
    #filename = "fluctuationsA"
    #handleIntegralData.writeDataFixedCutoff(cutoff, dArr, fluctuationsA, filename)
    #cutoff, dArr, fluctuationsA = handleIntegralData.retrieveDataFixedCutoff(filename)
    #baseFreqArr = np.pi * consts.c / dArr
    ##plotDosFP.plotFluctuationsAandE(dArr, fluctuationsE, fluctuationsA, baseFreqArr, cutoff)
    #plotDosFP.plotFluctuationsAandENaturalUnits(dArr, fluctuationsE, fluctuationsA, baseFreqArr, cutoff)
    #exit()


    #compute effective mass of free electrons and write it to a file

    #dArr = np.logspace(-6, -3, 20)
    #cutoff = 3. * 241.8 * 1e12
    #massArr = integratedDos.numericalIntegralEffectiveMass(cutoff, dArr)
    #filename = "delMassFP"
    #handleIntegralData.writeMassData(cutoff, dArr, massArr, filename)
    plotDosFP.plotEffectiveMassesComparison()

    exit()

    dArr = np.logspace(-6, -3, 20)
    cutoff = 3. * 241.8 * 1e12
    #filename = "delHoppingFP"
    #hopArr = integratedDos.numericalIntegralHopping(cutoff, dArr)
    #handleIntegralData.writeMassData(cutoff, dArr, hopArr, filename)
    plotDosFP.plotEffectiveHoppingComparison()


dosFPMain()
