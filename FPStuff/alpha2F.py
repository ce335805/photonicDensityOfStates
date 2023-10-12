import numpy as np


import computeAlpha2F
import plotAlpha2F

def computeAlpha2FMain():

    print("computing the alpha2F function for the FP cavity")


    #set parameters

    qArr = np.linspace(np.pi * 1e4, np.pi * 1e6, 200)
    #qArr = np.logspace(1, 9, 200)
    OmArr = np.array([0.])

    dArr = np.logspace(-6., -2., 5, endpoint=True)
    dArr = np.array([1e-6, 1e-5, 1e-4, 1e-2, 1e-1])
    print("dArr = {}".format(dArr))

    alphaCavTrans, alphaCavLong = computeAlpha2F.computeAlpha2FTransLongD(dArr, qArr, OmArr)
    print("alphaTrans.shape = {}".format(alphaCavTrans.shape))

    plotAlpha2F.plotLinesAsOfD(qArr, dArr, alphaCavTrans[:, :, 0], alphaCavLong[:, :, 0])

computeAlpha2FMain()