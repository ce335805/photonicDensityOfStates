from SphPStuff.wfFuncs import TEWavefunctionSPhP as TE, TEWavefunctionEvaSPhP as TEEva, TEWavefunctionResSPhP as TERes
from SphPStuff.wfFuncs import TMWavefunctionSPhP as TM
from SphPStuff.wfFuncs import TMWavefunctionEvaSPhP as TMEva
from SphPStuff.wfFuncs import TMWavefunctionResSPhP as TMRes
from SphPStuff.wfFuncs import TMWavefunctionSurf as Surf

from SphPStuff.dosFuncs import dosTEModes as dosTE
from SphPStuff.dosFuncs import dosTEEvaModes as dosTEEva
from SphPStuff.dosFuncs import dosTEResModes as dosTERes
from SphPStuff.dosFuncs import dosTMModes as dosTM
from SphPStuff.dosFuncs import dosTMEvaModes as dosTMEva
from SphPStuff.dosFuncs import dosTMResModes as dosTMRes
from SphPStuff.dosFuncs import dosTMSurfModes as dosTMSurf

from SphPStuff.asOfFrequency import dosAsOfFreq



def SPhPMain():
    print("Compute full Dos and all modes")

    #TE.createPlotTE()
    #TEEva.createPlotTEEva()
    #TERes.createPlotTERes()
    #TM.createPlotTM()
    #TMEva.createPlotTMEva()
    #TMRes.createPlotTMRes()
    #Surf.createPlotSurf()

    dosTE.createPlotDosTE()
    #dosTEEva.createPlotDosTEEva()
    #dosTERes.createPlotDosTERes()
    #dosTM.createPlotDosTM()
    #dosTMEva.createPlotDosTMEva()
    #dosTMRes.createPlotDosTMRes()
    #dosTMSurf.createPlotDosTMSurf()

    #dosAsOfFreq.createPlotAsOfOmega()

SPhPMain()