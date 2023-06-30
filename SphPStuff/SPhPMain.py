import numpy as np

import epsilonFunctions
import TEWavefunctionSPhP as TE
import TEWavefunctionEvaSPhP as TEEva
import TEWavefunctionResSPhP as TERes



def SPhPMain():
    print("Compute full Dos and all modes")

    TE.createPlotTE()
    TEEva.createPlotTEEva()
    TERes.createPlotTERes()

SPhPMain()