import numpy as np
import combinedDosPlots
import produceFreqData

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
    #epsInf = 6.3
    L = 1.
    zArr = np.logspace(-3, -9, 50, endpoint=True, base = 10)
    zArr = np.append(np.array([L / 4.]), zArr)

    produceFreqData.produceFreqIntegralData(zArr, wLO, wTO, epsInf, L)


    combinedDosPlots.plotDosWhole(zArr, wLO, wTO, epsInf, L)
    #combPlots.plotTMWhole()

    #dosAsOfFreq.createPlotAsOfOmega()
    #performFreqIntegral.freqIntegral()
    #plotTotalAsOfZ.dosTotalAsOfZ()


if __name__ == "__main__":
    main()