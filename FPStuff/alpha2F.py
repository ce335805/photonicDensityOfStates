import numpy as np


import computeAlpha2F
import plotAlpha2F

def computeAlpha2FMain():

    print("computing the alpha2F function for the FP cavity")


    #set parameters

    qArr = np.linspace(np.pi * 1e4, np.pi * 1e7, 100)
    OmArr = np.array([0.])

    dArr = np.logspace(-6., 0., 7, endpoint=True)
    print("dArr = {}".format(dArr))

    alphaCavTrans, alphaCavLong = computeAlpha2F.computeAlpha2FTransLongD(dArr, qArr, OmArr)
    print("alphaTrans.shape = {}".format(alphaCavTrans.shape))

    plotAlpha2F.plotLinesAsOfD(qArr, dArr, alphaCavTrans[:, :, 0], alphaCavLong[:, :, 0])

computeAlpha2FMain()