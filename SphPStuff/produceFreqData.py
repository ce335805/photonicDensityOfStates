from itertools import repeat
from multiprocessing import Pool
from time import perf_counter

# from asOfFrequency import dosAsOfFreq
import dosAsOfFreq as dosAsOfFreq
import h5py
import numpy as np


def parameterName(wmax, L, eps):
    wMaxStr = "wMax" + str(int(wmax * 1e-12))
    ellStr = "L" + str(int(10 * L))
    epsStr = "Eps" + str(int(10 * eps))
    return wMaxStr + ellStr + epsStr

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
    produceFreqDataAbove(arrAboveClose, zArr, wLO, wTO, epsInf, L)
    t_stop = perf_counter()
    print("Time to evaluate for frequency points above close: {}".format(t_stop - t_start))

    #produceFreqDataSurf(surfFreqArr, zArr, wLO, wTO, epsInf, L)

def defineFreqArrays(wLO, wTO, epsInf):
    arrBelow = np.linspace(wTO * 1e-1, wTO - 1e-3 * wTO, 100, endpoint=False)
    arrWithin = np.linspace(wTO, wLO, 100, endpoint=False)
    arrWithin = arrWithin[1:]
    #wEpsEq1 = np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1))
    arrAbove = np.linspace(wLO, 150. * wLO, 1500, endpoint=False)
    arrAbove = arrAbove[1:]

    #surfFreqArr = np.linspace(wTO, wLO, 500 + 1, endpoint=False)
    #surfFreqArr = surfFreqArr[1:]
    surfFreqArr = arrWithin

    return (arrBelow, arrWithin, arrAbove, surfFreqArr)

def produceFreqDataBelow(wArrBelow, zArr, wLO, wTO, epsInf, L):

    parameterStr = parameterName(wArrBelow[-1], L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArrBelow, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArrBelow, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataWithin(wArrWihtin, zArr, wLO, wTO, epsInf, L):

    parameterStr = parameterName(wArrWihtin[-1], L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataSurf(wArrWihtin, zArr, wLO, wTO, epsInf, L):
    filenameTE = 'savedData/dosSurf.h5'
    dosAsOfFreq.produceFreqDataSurf(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTE)

def produceFreqDataAbove(wArrAbove, zArr, wLO, wTO, epsInf, L):

    parameterStr = parameterName(wArrAbove[-1], L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArrAbove, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArrAbove, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataTE(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    with Pool() as pool:
        t_start = perf_counter()
        dosTEVals = np.array(pool.starmap(dosAsOfFreq.getDosTE, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        t_stop = perf_counter()
        #print("Time to evaluate TE: {}".format(t_stop - t_start))
        t_start = perf_counter()
        dosTEEvaVals = np.array(pool.starmap(dosAsOfFreq.getDosTEEva, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        t_stop = perf_counter()
        #print("Time to evaluate TEEva: {}".format(t_stop - t_start))
        t_start = perf_counter()
        dosTEResVals = np.array(pool.starmap(dosAsOfFreq.getDosTERes, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        t_stop = perf_counter()
        #print("Time to evaluate TERes: {}".format(t_stop - t_start))
    #dosTEVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTEEvaVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTEResVals = np.zeros((len(omegaArr), len(zArr)))
    #for omegaInd, omegaVal in enumerate(omegaArr):
    #   #print("omega = {}THz".format(omegaVal * 1e-12))
    #   dosTEVals[omegaInd, :] = dosAsOfFreq.getDosTE(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTEEvaVals[omegaInd, :] = dosAsOfFreq.getDosTEEva(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTEResVals[omegaInd, :] = dosAsOfFreq.getDosTERes(omegaVal, zArr, L, wLO, wTO, epsInf)
    ##   #print("")

    dosTETotal = 1. * dosTEVals + 1. * dosTEEvaVals + 1. * dosTEResVals
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTE', data=dosTETotal)
    h5f.close()

def produceFreqDataTM(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    with Pool() as pool:
        t_start = perf_counter()
        dosTMVals = np.array(pool.starmap(dosAsOfFreq.getDosTM, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        t_stop = perf_counter()
        #print("Time to evaluate TM: {}".format(t_stop - t_start))
        t_start = perf_counter()
        dosTMEvaVals = np.array(pool.starmap(dosAsOfFreq.getDosTMEva, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        t_stop = perf_counter()
        #print("Time to evaluate TM Eva: {}".format(t_stop - t_start))
        t_start = perf_counter()
        dosTMResVals = np.array(pool.starmap(dosAsOfFreq.getDosTMRes, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        t_stop = perf_counter()
        #print("Time to evaluate TMRes: {}".format(t_stop - t_start))
        #dosTMSurfVals = np.array(pool.starmap(dosAsOfFreq.getDosTMSurf, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))

    #dosTMVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTMEvaVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTMResVals = np.zeros((len(omegaArr), len(zArr)))
    #dosTMSurfVals = np.zeros((len(omegaArr), len(zArr)))
    #for omegaInd, omegaVal in enumerate(omegaArr):
    #   #print("omega = {}THz".format(omegaVal * 1e-12))
    #   dosTMVals[omegaInd, :] = dosAsOfFreq.getDosTM(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTMEvaVals[omegaInd, :] = dosAsOfFreq.getDosTMEva(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   dosTMResVals[omegaInd, :] = dosAsOfFreq.getDosTMRes(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   #dosTMSurfVals[omegaInd, :] = dosAsOfFreq.getDosTMSurf(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   #print("")

    dosTMTotal = 1. * dosTMVals + 1. * dosTMEvaVals + 1. * dosTMResVals# + dosTMSurfVals
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTM', data=dosTMTotal)
    h5f.close()


def retrieveDosTE(wMaxBelow, wMaxWithin, wMAxAbove, L, epsInf):

    parameterStr = parameterName(wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = h5f['dosTE'][:]
    h5f.close()

    parameterStr = parameterName(wMaxWithin, L, epsInf)

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:], axis = 0)
    h5f.close()

    parameterStr = parameterName(wMAxAbove, L, epsInf)

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:], axis = 0)
    h5f.close()

    return dosTETotal

def retrieveDosTM(wMaxBelow, wMaxWithin, wMAxAbove, L, epsInf):

    parameterStr = parameterName(wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = h5f['dosTM'][:]
    h5f.close()

    parameterStr = parameterName(wMaxWithin, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = np.append(dosTMTotal, h5f['dosTM'][:], axis = 0)
    h5f.close()

    parameterStr = parameterName(wMAxAbove, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
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
