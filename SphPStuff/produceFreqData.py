from itertools import repeat
from multiprocessing import Pool
from time import perf_counter

# from asOfFrequency import dosAsOfFreq
import dosAsOfFreq as dosAsOfFreq
import h5py
import numpy as np


def parameterName(wLO, wTO, wmax, L, eps):
    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    wMaxStr = "wMax" + str(int(wmax * 1e-12))
    ellStr = "L" + str(int(10 * L))
    epsStr = "Eps" + str(int(10 * eps))
    return wMaxStr + ellStr + epsStr + wLOStr + wTOStr

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
    arrBelow = np.linspace(wTO * 1e-6, wTO, 100, endpoint=False)
    arrWithin = np.linspace(wTO, wLO, 200, endpoint=False)
    arrWithin = arrWithin[1:]
    #arrAbove1 = np.linspace(wLO, 50. * wLO, 5000, endpoint=False)
    #arrAbove2 = np.linspace(50. * wLO, 99. * wLO, 1000, endpoint=False)
    #arrAbove2 = arrAbove2[1:]
    #arrAbove = np.append(arrAbove1, arrAbove2)
    #arrAbove = np.linspace(wLO, 300 * 1e12, 2000, endpoint=False)
    arrAbove = np.linspace(wLO, 100 * 1e12, 400, endpoint=False)
    arrAbove = arrAbove[1:]

    #parameters for real-space plot
    #arrBelow = np.linspace(wTO * 1e-1, wTO - 1e-3 * wTO, 5, endpoint=False)
    #arrWithin = np.linspace(wTO, wLO, 5, endpoint=False)
    #arrWithin = arrWithin[1:]
    #arrAbove = np.linspace(wLO, 2. * wLO, 5, endpoint=False)
    #arrAbove = arrAbove[1:]


    surfFreqArr = arrWithin

    return (arrBelow, arrWithin, arrAbove, surfFreqArr)

def produceFreqDataBelow(wArrBelow, zArr, wLO, wTO, epsInf, L):

    parameterStr = parameterName(wLO, wTO, wArrBelow[-1], L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArrBelow, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArrBelow, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataWithin(wArrWihtin, zArr, wLO, wTO, epsInf, L):

    parameterStr = parameterName(wLO, wTO, wArrWihtin[-1], L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataSurf(wArrWihtin, zArr, wLO, wTO, epsInf, L):
    filenameTE = 'savedData/dosSurf.h5'
    dosAsOfFreq.produceFreqDataSurf(wArrWihtin, zArr, L, wLO, wTO, epsInf, filenameTE)

def produceFreqDataAbove(wArrAbove, zArr, wLO, wTO, epsInf, L):

    parameterStr = parameterName(wLO, wTO, wArrAbove[-1], L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArrAbove, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArrAbove, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataTE(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    with Pool() as pool:
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
    ##   #print("")

    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTE', data=dosTEVals)
    h5f.create_dataset('dosTEEva', data=dosTEEvaVals)
    h5f.create_dataset('dosTERes', data=dosTEResVals)
    h5f.close()

def produceFreqDataTM(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    with Pool() as pool:

        #dosTMVals = np.array(pool.starmap(dosAsOfFreq.getDosTM, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        retTM = np.array(pool.starmap(dosAsOfFreq.getDosTMParaPerp, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        #dosTMEvaVals = np.array(pool.starmap(dosAsOfFreq.getDosTMEva, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        retTMEva = np.array(pool.starmap(dosAsOfFreq.getDosTMEvaParaPerp, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        #dosTMResVals = np.array(pool.starmap(dosAsOfFreq.getDosTMRes, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
        retTMRes = np.array(pool.starmap(dosAsOfFreq.getDosTMResParaPerp, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))


    dosTMValsPara = retTM[:, 0, :]
    dosTMValsPerp = retTM[:, 1, :]
    dosTMEvaValsPara = retTMEva[:, 0, :]
    dosTMEvaValsPerp = retTMEva[:, 1, :]
    dosTMResValsPara = retTMRes[:, 0, :]
    dosTMResValsPerp = retTMRes[:, 1, :]

    #dosTMValsPara = np.zeros((len(omegaArr), len(zArr)))
    #dosTMValsPerp = np.zeros((len(omegaArr), len(zArr)))
    #dosTMEvaValsPara = np.zeros((len(omegaArr), len(zArr)))
    #dosTMEvaValsPerp = np.zeros((len(omegaArr), len(zArr)))
    #dosTMResValsPara = np.zeros((len(omegaArr), len(zArr)))
    #dosTMResValsPerp = np.zeros((len(omegaArr), len(zArr)))
    #for omegaInd, omegaVal in enumerate(omegaArr):
    #   #print("omega = {}THz".format(omegaVal * 1e-12))
    #   (dosTMValsPara[omegaInd, :], dosTMValsPerp[omegaInd, :]) = dosAsOfFreq.getDosTMParaPerp(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   (dosTMEvaValsPara[omegaInd, :], dosTMEvaValsPerp[omegaInd, :]) = dosAsOfFreq.getDosTMEvaParaPerp(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   (dosTMResValsPara[omegaInd, :], dosTMResValsPerp[omegaInd, :]) = dosAsOfFreq.getDosTMResParaPerp(omegaVal, zArr, L, wLO, wTO, epsInf)
    #   #print("")

    dosTMParaTotal = dosTMValsPara + dosTMEvaValsPara + dosTMResValsPara
    dosTMPerpTotal = dosTMValsPerp + dosTMEvaValsPerp + dosTMResValsPerp

    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTMPara', data=dosTMValsPara)
    h5f.create_dataset('dosTMEvaPara', data=dosTMEvaValsPara)
    h5f.create_dataset('dosTMResPara', data=dosTMResValsPara)
    h5f.create_dataset('dosTMPerp', data=dosTMValsPerp)
    h5f.create_dataset('dosTMEvaPerp', data=dosTMEvaValsPerp)
    h5f.create_dataset('dosTMResPerp', data=dosTMResValsPerp)
    h5f.close()


def retrieveDosTE(wMaxBelow, wMaxWithin, wMAxAbove, L, wLO, wTO, epsInf):

    parameterStr = parameterName(wLO, wTO, wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:]
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMaxWithin, L, epsInf)

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:], axis = 0)
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMAxAbove, L, epsInf)

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:], axis = 0)
    h5f.close()

    return dosTETotal

def retrieveDosTEMultipleWMax(wMaxBelow, wMaxWithin, wMaxAboveArr, L, wLO, wTO, epsInf):

    parameterStr = parameterName(wLO, wTO, wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:]
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMaxWithin, L, epsInf)

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = np.append(dosTETotal, h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:], axis = 0)
    h5f.close()

    for wMaxAbove in wMaxAboveArr:

        parameterStr = parameterName(wLO, wTO, wMaxAbove, L, epsInf)

        filenameTE = dir + 'dosTE' + parameterStr + '.h5'
        h5f = h5py.File(filenameTE, 'r')
        dosTETotal = np.append(dosTETotal, h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:], axis = 0)
        h5f.close()

    return dosTETotal

def retrieveDosTMTotal(wMaxBelow, wMaxWithin, wMAxAbove, L, wLO, wTO, epsInf):

    parameterStr = parameterName(wLO, wTO, wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:] + h5f['dosTMPerp'][:] + h5f['dosTMEvaPerp'][:] + h5f['dosTMResPerp'][:]
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMaxWithin, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = np.append(dosTMTotal, h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:] + h5f['dosTMPerp'][:] + h5f['dosTMEvaPerp'][:] + h5f['dosTMResPerp'][:], axis = 0)
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMAxAbove, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = np.append(dosTMTotal, h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:] + h5f['dosTMPerp'][:] + h5f['dosTMEvaPerp'][:] + h5f['dosTMResPerp'][:], axis = 0)
    h5f.close()

    return dosTMTotal


def retrieveDosTMPara(wMaxBelow, wMaxWithin, wMAxAbove, L, wLO, wTO, epsInf):

    parameterStr = parameterName(wLO, wTO, wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMPara = h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:]
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMaxWithin, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMPara = np.append(dosTMPara, h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:], axis = 0)
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMAxAbove, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMPara = np.append(dosTMPara, h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:], axis = 0)
    h5f.close()

    return dosTMPara

def retrieveDosTMParaMultipleWMax(wMaxBelow, wMaxWithin, wMaxAboveArr, L, wLO, wTO, epsInf):

    parameterStr = parameterName(wLO, wTO, wMaxBelow, L, epsInf)

    dir = "savedData/clusterFreqData/"
    #dir = "savedData/"

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:]
    h5f.close()

    parameterStr = parameterName(wLO, wTO, wMaxWithin, L, epsInf)

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = np.append(dosTMTotal, h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:], axis = 0)
    h5f.close()

    for wMaxAbove in wMaxAboveArr:

        parameterStr = parameterName(wLO, wTO, wMaxAbove, L, epsInf)

        filenameTM = dir + 'dosTM' + parameterStr + '.h5'
        h5f = h5py.File(filenameTM, 'r')
        dosTMTotal = np.append(dosTMTotal, h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:], axis = 0)
        h5f.close()

    return dosTMTotal

def retrieveDosSurf():
    filename = 'savedData/dosSurf.h5'
    h5f = h5py.File(filename, 'r')
    dosSurf = h5f['dosTM'][:]
    h5f.close()

    return dosSurf

def writeMasses(cutoff, zArr, mArr, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('cutoff', data=np.array([cutoff]))
    h5f.create_dataset('zArr', data=zArr)
    h5f.create_dataset('delM', data=mArr)
    h5f.close()

def readMasses(filename):
    h5f = h5py.File(filename, 'r')
    cutoff = h5f['cutoff'][:]
    mArr = h5f['delM'][:]
    zArr = h5f['zArr'][:]
    h5f.close()
    return (cutoff, zArr, mArr)
