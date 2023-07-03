from SphPStuff.wfFuncs import TEWavefunctionSPhP as TE, TEWavefunctionEvaSPhP as TEEva, TEWavefunctionResSPhP as TERes
from SphPStuff.wfFuncs import TMWavefunctionSPhP as TM


from SphPStuff.dosFuncs import dosTEModes as dosTE
from SphPStuff.dosFuncs import dosTEEvaModes as dosTEEva
from SphPStuff.dosFuncs import dosTEResModes as dosTERes

def SPhPMain():
    print("Compute full Dos and all modes")

    #TE.createPlotTE()
    #TEEva.createPlotTEEva()
    #TERes.createPlotTERes()
    TM.createPlotTM()

    #dosTE.createPlotDosTE()
    #dosTEEva.createPlotDosTEEva()
    #dosTERes.createPlotDosTERes()


SPhPMain()