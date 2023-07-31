from wfFuncs import TEWavefunctionSPhP as TE, TEWavefunctionEvaSPhP as TEEva, TEWavefunctionResSPhP as TERes
from wfFuncs import TMWavefunctionSPhP as TM
from wfFuncs import TMWavefunctionSPhP as TM
from wfFuncs import TMWavefunctionEvaSPhP as TMEva
from wfFuncs import TMWavefunctionResSPhP as TMRes
from wfFuncs import TMWavefunctionSurf as Surf

from dosFuncs import dosTEModes as dosTE
from dosFuncs import dosTEEvaModes as dosTEEva
from dosFuncs import dosTEResModes as dosTERes
from dosFuncs import dosTMModes as dosTM
from dosFuncs import dosTMEvaModes as dosTMEva
from dosFuncs import dosTMResModes as dosTMRes
from dosFuncs import dosTMSurfModes as dosTMSurf
from dosFuncs import combinedDosPlots as combPlots

from asOfFrequency import dosAsOfFreq
from asOfFrequency import performFreqIntegral
from plotTotalAsOfZ import dosTotalAsOfZ


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

    #combPlots.plotTEWhole()
    #combPlots.plotTMWhole()

    #dosAsOfFreq.createPlotAsOfOmega()
    performFreqIntegral.freqIntegral()
    #dosTotalAsOfZ()


if __name__ == "__main__":
    main()