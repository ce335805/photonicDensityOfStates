import numpy as np

import epsilonFunctions
import TEWavefunctionSPhP as TE
import TEWavefunctionEvaSPhP as TEEva
import TEWavefunctionResSPhP as TERes



def SPhPMain():
    print("Compute full Dos and all modes")

    #epsInf = 2.
    #wTO = 1. * 1e12
    #wLO = 3. * 1e12
    #omegaArr = np.linspace(0., 1e13, 500)
    #epsArr = epsilonFunctions.epsilon(omegaArr, wLO, wTO, epsInf)
    #epsilonFunctions.plotEpsilon(omegaArr, epsArr, epsInf, wLO, wTO)

    #TE.createPlotTE()
    #TEEva.createPlotTEEva()
    TERes.createPlotTERes()

SPhPMain()