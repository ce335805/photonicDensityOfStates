import numpy as np
import h5py

import produceFreqData
import plotAsOfZ as plotAsOfZ

def dosTotalAsOfZ():
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = 1.
    zArr = np.linspace(- 2. * L * 1e-3, 2. * L * 1e-3, 500)

    #produceZData(zArr, wLO, wTO, epsInf, L)
    producePlotAsOfZ(zArr, wLO, wTO, epsInf, L)

def produceZData(zArr, wLO, wTO, epsInf, L):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    omegaArr = np.array([.5 * 1e12, 1.2 * 1e12, wInf - 1e-2 * wInf, 2.5 * 1e12, 3.5 * 1e12, 5 * 1e12])
    produceZArrData(omegaArr, zArr, wLO, wTO, epsInf, L)

def produceZArrData(wArr, zArr, wLO, wTO, epsInf, L):

    filenameTE = 'savedData/dosTEAsOfZDielectric.h5'
    produceFreqData.produceFreqDataTE(wArr, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTMAsOfZDielectric.h5'
    produceFreqData.produceFreqDataTM(wArr, zArr, L, wLO, wTO, epsInf, filenameTM)


def producePlotAsOfZ(zArr, wLO, wTO, epsInf, L):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    omegaArr = np.array([.5 * 1e12, 1.2 * 1e12, wInf - 1e-2 * wInf, 2.5 * 1e12, 3.5 * 1e12, 5 * 1e12])

    filenameTE = 'savedData/dosTEAsOfZDielectric.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = h5f['dosTE'][:]
    h5f.close()

    filenameTM = 'savedData/dosTMAsOfZDielectric.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = h5f['dosTM'][:]
    h5f.close()

    filename = "TE"
    plotAsOfZ.plotDosAsOfFreqDosTotal(dosTETotal, zArr, L, omegaArr, wLO, wTO, epsInf, filename)

    filename = "TM"
    plotAsOfZ.plotDosAsOfFreqDosTotal(dosTMTotal, zArr, L, omegaArr, wLO, wTO, epsInf, filename)
