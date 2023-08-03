from itertools import repeat
from multiprocessing import Pool
from time import perf_counter

# from asOfFrequency import dosAsOfFreq
import dosAsOfFreq as dosAsOfFreq
import h5py
import numpy as np


def produceFreqIntegralData(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAboveClose, surfFreqArr = defineFreqArrays(wLO, wTO, epsInf)

    t_start = perf_counter()
    produceFreqDataBelow(arrBelow, zArr, wLO, wTO, epsInf, L)
    t_stop = perf_counter()
    print("Time to evaluate for frequency points below: {}".format(t_stop - t_start))

    t_start = perf_counter()
    produceFreqDataWithin(arrWithin, zArr, wLO, wTO, epsInf, L)
    t_stop = perf_counter()
    print("Time to evaluate for frequency points within: {}".format(t_stop - t_start))

    t_start = perf_counter()
    produceFreqDataAboveClose(arrAboveClose, zArr, wLO, wTO, epsInf, L)
    t_stop = perf_counter()
    print("Time to evaluate for frequency points above close: {}".format(t_stop - t_start))

    #produceFreqDataSurf(surfFreqArr, zArr, wLO, wTO, epsInf, L)

def defineFreqArrays(wLO, wTO, epsInf):
    arrBelow = np.linspace(wTO * 1e-1, wTO - 1e-3 * wTO, 100, endpoint=False)
    arrWithin = np.linspace(wTO, wLO, 100, endpoint=False)
    arrWithin = arrWithin[1:]
    #wEpsEq1 = np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1))
    arrAbove = np.linspace(wLO, 10. * wLO, 300, endpoint=False)
    arrAbove = arrAbove[1:]

    #surfFreqArr = np.linspace(wTO, wLO, 500 + 1, endpoint=False)
    #surfFreqArr = surfFreqArr[1:]
    surfFreqArr = arrWithin

    return (arrBelow, arrWithin, arrAbove, surfFreqArr)

def produceFreqDataBelow(wArrBelow, zArr, wLO, wTO, epsInf, L):

    filenameTE = 'savedData/dosTEBelow.h5'
    produceFreqDataTE(wArrBelow, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTMBelow.h5'
    produceFreqDataTM(wArrBelow, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataWithin(wArrWihtin, zArr, wLO, wTO, epsInf, L):

    filenameTE = 'savedData/dosTEWithin.h5'
    produceFreqDataTE(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTMWithin.h5'
    produceFreqDataTM(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataSurf(wArrWihtin, zArr, wLO, wTO, epsInf, L):
    filenameTE = 'savedData/dosSurf.h5'
    dosAsOfFreq.produceFreqDataSurf(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTE)

def produceFreqDataAboveClose(wArrAbove, zArr, wLO, wTO, epsInf, L):

    filenameTE = 'savedData/dosTEAbove.h5'
    produceFreqDataTE(wArrAbove, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTMAbove.h5'
    produceFreqDataTM(wArrAbove, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataTE(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    with Pool(1) as pool:
        dosTEVals = np.array(pool.starmap(dosAsOfFreq.getDosTE, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        dosTEEvaVals = np.array(pool.starmap(dosAsOfFreq.getDosTEEva, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        dosTEResVals = np.array(pool.starmap(dosAsOfFreq.getDosTERes, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))

    #dosTEVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTEEvaVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTEResVals = np.zeros((len(omegaArr), len(zArr)))
    #for omegaInd, omegaVal in enumerate(omegaArr):
    #   #print("omega = {}THz".format(omegaVal * 1e-12))
    #   dosTEVals[omegaInd, :] = dosAsOfFreq.getDosTE(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTEEvaVals[omegaInd, :] = dosAsOfFreq.getDosTEEva(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTEResVals[omegaInd, :] = dosAsOfFreq.getDosTERes(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   #print("")

    dosTETotal = dosTEVals + dosTEEvaVals + dosTEResVals
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTE', data=dosTETotal)
    h5f.close()

def produceFreqDataTM(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    with Pool(1) as pool:
        dosTMVals = np.array(pool.starmap(dosAsOfFreq.getDosTM, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        dosTMEvaVals = np.array(pool.starmap(dosAsOfFreq.getDosTMEva, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        dosTMResVals = np.array(pool.starmap(dosAsOfFreq.getDosTMRes, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        dosTMSurfVals = np.array(pool.starmap(dosAsOfFreq.getDosTMSurf, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))

    #dosTMVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTMEvaVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTMResVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTMSurfVals = np.zeros((len(omegaArr), len(zArr)))
    #for omegaInd, omegaVal in enumerate(omegaArr):
    #   #print("omega = {}THz".format(omegaVal * 1e-12))
    #   dosTMVals[omegaInd, :] = dosAsOfFreq.getDosTM(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTMEvaVals[omegaInd, :] = dosAsOfFreq.getDosTMEva(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTMResVals[omegaInd, :] = dosAsOfFreq.getDosTMRes(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTMSurfVals[omegaInd, :] = dosAsOfFreq.getDosTMSurf(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   #print("")

    dosTMTotal = dosTMVals + dosTMEvaVals + dosTMResVals + dosTMSurfVals
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTM', data=dosTMTotal)
    h5f.close()


def retrieveDosTE():

    dir = "savedData/clusterFreqDataNoSPhP/"

    filenameTE = dir + 'dosTEBelow.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = h5f['dosTE'][:]
    h5f.close()

    filenameTE = dir + 'dosTEWithin.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:], axis = 0)
    h5f.close()

    filenameTE = dir + 'dosTEAbove.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:], axis = 0)
    h5f.close()

    return dosTETotal

def retrieveDosTM():

    dir = "savedData/clusterFreqDataNoSPhP/"

    filenameTM = dir + 'dosTMBelow.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = h5f['dosTM'][:]
    h5f.close()

    filenameTM = dir + 'dosTMWithin.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = np.append(dosTMTotal, h5f['dosTM'][:], axis = 0)
    h5f.close()

    filenameTM = dir + 'dosTMAbove.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = np.append(dosTMTotal, h5f['dosTM'][:], axis = 0)
    h5f.close()

    return dosTMTotal

def retrieveDosSurf():
    filename = 'savedData/dosSurf.h5'
    h5f = h5py.File(filename, 'r')
    dosSurf = h5f['dosTM'][:]
    h5f.close()

    return dosSurf
