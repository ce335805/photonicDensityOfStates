import numpy as np
import epsilonFunctions



def SPhPMain():
    print("Compute full Dos and all modes")

    epsInf = 2.
    wTO = 1. * 1e12
    wLO = 3. * 1e12
    omegaArr = np.linspace(0., 1e13, 500)
    epsArr = epsilonFunctions.epsilon(omegaArr, wLO, wTO, epsInf)
    print("epsArr.shape = {}".format(epsArr.shape))
    epsilonFunctions.plotEpsilon(omegaArr, epsArr, epsInf, wLO, wTO)

SPhPMain()