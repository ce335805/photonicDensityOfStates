import numpy as np
import combinedDosPlots
import produceFreqData
import performFreqIntegral
import thesisPlots
import scipy.constants as consts
import massScalingPlots

def main():
    print("Compute full Dos and all modes")

    #thesisPlots.plotEpsilonOmega()
    #thesisPlots.plotDispersion()
    #exit()

    #TE.createPlotTE()
    #TEEva.createPlotTEEva()
    #TERes.createPlotTERes()
    #TM.createPlotTM()
    #TMEva.createPlotTMEva()
    #TMRes.createPlotTMRes()
    #Surf.createPlotSurf()

    #dosTE.createPlotDosTE()
    #dosTEEva.createPlotDosTEEva()
    #dosTERes.createPlotDosTERes()
    #dosTM.createPlotDosTM()
    #dosTMEva.createPlotDosTMEva()
    #dosTMRes.createPlotDosTMRes()
    #dosTMSurf.createPlotDosTMSurf()

    epsInf = 1.
    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    #wLO = 20. * 1e12
    wTO = 1e6
    L = 1.
    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf
    zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), 1000, endpoint=True, base = 10)
    #zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-3 * lambda0), 300, endpoint=True, base = 10)
    zArr = np.append([L / 4.], zArr)

    #parameters for real-space plot
    #wLO = 32.04 * 1e12
    #wTO = 7.92 * 1e12
    #L = .01
    #zArr = np.linspace(-L / 2., L / 2., 10000)

    #produceFreqData.produceFreqIntegralData(zArr, wLO, wTO, epsInf, L)

    #wLOArr = np.array([2., 5., 10., 20.]) * 1e12
    #wTOArr = np.array([1., 1., 1., 1.]) * 1e6
    #for wInd, _ in enumerate(wLOArr):
    #    produceFreqData.produceFreqIntegralData(zArr, wLOArr[wInd], wTOArr[wInd], epsInf, L)

    #cutoff = 1e15
    #performFreqIntegral.produceCollapsePlotE(zArr, cutoff, wLOArr, wTOArr, epsInf, L)
    #performFreqIntegral.produceCollapsePlotA(zArr, cutoff, wLOArr, wTOArr, epsInf, L)

    combinedDosPlots.plotDosWhole(zArr, wLO, wTO, epsInf, L)
    #performFreqIntegral.freqIntegral(zArr, wLO, wTO, epsInf, L)
    #performFreqIntegral.computeSumRuleMultipleCutoffs(zArr, wLO, wTO, epsInf, L)
    #performFreqIntegral.computeFluctuationsMultipleCutoffs(zArr, wLO, wTO, epsInf, L)

    #massScalingPlots.plotMassScaling(L)

if __name__ == "__main__":
    main()