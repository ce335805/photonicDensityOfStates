import numpy as np
import combinedDosPlots
import produceFreqData
import performFreqIntegral

def main():
    print("Compute full Dos and all modes")

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

    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 1.
    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    wLO = 20. * 1e12
    wTO = 1e6
    L = 0.01
    zArr = np.logspace(-3, -9, 50, endpoint=True, base = 10)
    zArr = np.append(np.array([L / 4.]), zArr)

    #produceFreqData.produceFreqIntegralData(zArr, wLO, wTO, epsInf, L)

    wLOArr = np.array([2., 5., 10., 20.]) * 1e12
    wTOArr = np.array([1., 1., 1., 1.]) * 1e6
    #for wInd, _ in enumerate(wLOArr):
    #    produceFreqData.produceFreqIntegralData(zArr, wLOArr[wInd], wTOArr[wInd], epsInf, L)

    cutoff = 1e14
    performFreqIntegral.produceCollapsePlot(zArr, cutoff, wLOArr, wTOArr, epsInf, L)

    #combinedDosPlots.plotDosWhole(zArr, wLO, wTO, epsInf, L)
    #performFreqIntegral.freqIntegral(zArr, wLO, wTO, epsInf, L)
    #performFreqIntegral.computeSumRuleMultipleCutoffs(zArr, wLO, wTO, epsInf, L)
    #performFreqIntegral.computeFluctuationsMultipleCutoffs(zArr, wLO, wTO, epsInf, L)

if __name__ == "__main__":
    main()