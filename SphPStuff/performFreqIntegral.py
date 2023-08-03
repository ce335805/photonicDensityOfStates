import numpy as np
import h5py
import scipy.constants as consts
import scipy.integrate

import epsilonFunctions as epsFunc

from dosFuncs import dosTEModes as dosTE
from dosFuncs import dosTEEvaModes as dosTEEva
from dosFuncs import dosTEResModes as dosTERes
from dosFuncs import dosTMModes as dosTM
from dosFuncs import dosTMEvaModes as dosTMEva
from dosFuncs import dosTMResModes as dosTMRes
from dosFuncs import dosTMSurfModes as dosTMSurf

import plotAsOfFreq as plotFreq
import dosAsOfFreq
import produceFreqData as prod

def freqIntegral():
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 1.
    L = 10.
    zArr = np.logspace(np.log10(L / 4), -10, 50, endpoint=True, base = 10)
    #zArr = np.logspace(-3, -9, 100)

    #intNum = performSPhPIntegralNum(1. * 1e-8, wLO, wTO, epsInf)
    #print("intNum = {}".format(intNum[0] * 1e-20))
    #print("intNum Err = {}".format(intNum[1] * 1e-20))
#
    #dosAna, dosNum = computeSPhPIntAsOfZ(zArr, wLO, wTO, epsInf)
    #plotFreq.compareSPhPInt(dosAna * 1e-20, dosNum * 1e-20, zArr, "SPhPComp")

    prod.produceFreqIntegralData(zArr, wLO, wTO, epsInf, L)
    producePlotAsOfFreq(zArr, wLO, wTO, epsInf, L)
    #computeFreqIntegralAsOfCutoff(zArr, wLO, wTO, epsInf, L)


def computeFreqIntegralAsOfCutoff(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAboveClose, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAboveClose)

    dosTETotal = prod.retrieveDosTE()
    dosTMTotal = prod.retrieveDosTM()
    dosSurf = prod.retrieveDosSurf()



    dosIntTE = np.zeros(dosTETotal.shape)
    dosIntTM = np.zeros(dosTMTotal.shape)
    dosIntSurf = np.zeros((len(arrWithin), len(zArr)))
    latConst = 1e-10

    #integrate surface part separately on finer grid
    #for zInd, zVal in enumerate(zArr):
    #    for wInd, wVal in enumerate(arrWithin):
    #        wInds = np.where(surfFreqArr < wVal)[0]
    #        wSurfPart = surfFreqArr[wInds]
    #        prefac = 2. * consts.fine_structure / 3. / np.pi * latConst ** 2 / consts.c ** 2 * wSurfPart
    #        intFuncSurf = prefac * dosSurf[wInds, zInd]
    #        dosIntSurf[wInd, zInd] = np.trapz(intFuncSurf, wSurfPart, axis=0)

    #patching
    #patchBelow = np.zeros((len(arrBelow), len(zArr)))
    #patchAbove = np.ones((len(arrAboveClose) + len(arrAboveFar), len(zArr)))
    #for wInd in range(len(arrAboveClose) + len(arrAboveFar)):
    #    patchAbove[wInd, :] = patchAbove[wInd, :] * dosIntSurf[-1, :]
    #dosIntSurf = np.append(patchBelow, dosIntSurf, axis = 0)
    #dosIntSurf = np.append(dosIntSurf, patchAbove, axis = 0)

    for zInd, zVal in enumerate(zArr):
        for wInd, wVal in enumerate(wArr):
            wArrPart = wArr[:wInd] * 1e-12
            #prefac = 2. * consts.fine_structure / np.pi * latConst ** 2 / consts.c ** 2 * wArrPart
            prefacFieldStrength = consts.hbar * wArrPart**1 / (2 * np.pi**2 * consts.epsilon_0 * consts.c**3)
            prefacFieldStrength = wArrPart ** 3
            #intFuncTE = prefacFieldStrength * (dosTETotal[:wInd, zInd] - .5)
            intFuncTE = prefacFieldStrength * (dosTETotal[:wInd, zInd] - dosTETotal[:wInd, 0])
            dosIntTE[wInd, zInd] = np.trapz(intFuncTE, x=wArrPart, axis = 0)
            #intFuncTM = prefacFieldStrength * (dosTMTotal[:wInd, zInd] - .5)
            intFuncTM = prefacFieldStrength * (dosTMTotal[:wInd, zInd] - dosTMTotal[:wInd, 0])
            dosIntTM[wInd, zInd] = np.trapz(intFuncTM, x=wArrPart, axis = 0)

    #dosIntTM = dosIntTM + dosIntSurf

    filename = "TEField"
    plotFreq.plotDosIntegratedAsOfCutoff(dosIntTE, zArr, L, wArr, wLO, wTO, epsInf, filename)
    filename = "TMField"
    plotFreq.plotDosIntegratedAsOfCutoff(dosIntTM, zArr, L, wArr, wLO, wTO, epsInf, filename)

def computeLocalFieldStrength(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAboveClose, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAboveClose)

    dosTETotal = prod.retrieveDosTE()
    dosTMTotal = prod.retrieveDosTM()
    dosSurf = prod.retrieveDosSurf()

    dosIntTE = np.zeros(dosTETotal.shape)
    dosIntTM = np.zeros(dosTMTotal.shape)
    dosIntSurf = np.zeros((len(arrWithin), len(zArr)))
    latConst = 1e-10

    for zInd, zVal in enumerate(zArr):
        for wInd, wVal in enumerate(arrWithin):
            wInds = np.where(surfFreqArr < wVal)[0]
            wSurfPart = surfFreqArr[wInds]
            prefac = consts.hbar / 2. / consts.epsilon_0
            intFuncSurf = prefac * dosSurf[wInds, zInd]
            dosIntSurf[wInd, zInd] = np.trapz(intFuncSurf, wSurfPart, axis=0)

    patchBelow = np.zeros((len(arrBelow), len(zArr)))
    patchAbove = np.ones((len(arrAboveClose), len(zArr)))
    for wInd in range(len(arrAboveClose)):
        patchAbove[wInd, :] = patchAbove[wInd, :] * dosIntSurf[-1, :]
    dosIntSurf = np.append(patchBelow, dosIntSurf, axis = 0)
    dosIntSurf = np.append(dosIntSurf, patchAbove, axis = 0)

    for zInd, zVal in enumerate(zArr):
        for wInd, wVal in enumerate(wArr):
            wArrPart = wArr[:wInd]
            prefac = 2. * consts.fine_structure / 3. / np.pi * latConst ** 2 / consts.c ** 2 * wArrPart
            intFuncTE = prefac * (dosTETotal[:wInd, zInd] - .5)
            dosIntTE[wInd, zInd] = np.trapz(intFuncTE, wArrPart, axis = 0)
            intFuncTM = prefac * (dosTMTotal[:wInd, zInd] - .5)
            dosIntTM[wInd, zInd] = np.trapz(intFuncTM, wArrPart, axis = 0)

    dosIntTM = dosIntTM + dosIntSurf

    filename = "TEInt"
    plotFreq.plotDosAsOfFreqDosTotal(dosIntTE, zArr, L, wArr, wLO, wTO, epsInf, filename)
    filename = "TMInt"
    plotFreq.plotDosAsOfFreqDosTotal(dosIntTM, zArr, L, wArr, wLO, wTO, epsInf, filename)



def producePlotAsOfFreq(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAboveClose, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAboveClose)

    dosTETotal = prod.retrieveDosTE()
    dosTMTotal = prod.retrieveDosTM()

    #dosSurf = prod.retrieveDosSurf()
    #dosSurf = patchDosSurfWithZeros(dosSurf, zArr, arrBelow, arrAboveClose, arrAboveFar)
    #dosTMTotal = dosTMTotal + dosSurf

    filename = "TE"
    plotFreq.plotDosAsOfFreqDosTotal(dosTETotal, zArr, L, wArr, wLO, wTO, epsInf, filename)
    filename = "TM"
    plotFreq.plotDosAsOfFreqDosTotal(dosTMTotal, zArr, L, wArr, wLO, wTO, epsInf, filename)
    #plotFreq.plotDosTotalWithSurfExtra(dosTMTotal, dosSurf, zArr, L, wArr, surfFreqArr, wLO, wTO, epsInf, filename)


def intFuncEffectiveHopping(omega, zVal, wLO, wTO, epsInf):
    prefac = 2. * consts.fine_structure / np.pi / consts.c ** 2 * omega # / 3.
    return prefac * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncFieldStrength(omega, zVal, wLO, wTO, epsInf):
    prefac = consts.hbar / 2. / consts.fine_structure
    return prefac * omega**3 * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def performSPhPIntegralNum(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    return scipy.integrate.quad(intFuncFieldStrength, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 500)
    #return scipy.integrate.quad(intFuncEffectiveHopping, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 500)

def performSPhPIntegralAna(zVal, wLO, wTO, epsInf):
    prefac = 2. * np.pi * consts.fine_structure * consts.c
    num = epsInf* np.sqrt(1 + epsInf) * (wLO**2 - wTO**2)
    denom = 4. * np.pi * np.sqrt(epsInf * wLO**2 + wTO**2) * (2. * epsInf * wLO**2 + (1 + epsInf**2) * wTO**2) * zVal**3
    return prefac * num / denom

def computeSPhPIntAsOfZ(zArr, wLO, wTO, epsInf):
    intAna = performSPhPIntegralAna(zArr, wLO, wTO, epsInf)
    intNum = np.zeros(intAna.shape)
    for zInd, zVal in enumerate(zArr):
        intNum[zInd] = performSPhPIntegralNum(zVal, wLO, wTO, epsInf)[0]

    return (intAna, intNum)

def patchDosSurfWithZeros(dosSurf, zArr, arrBelow, arrAboveClose):
    patchBelow = np.zeros((len(arrBelow), len(zArr)))
    patchAbove = np.zeros((len(arrAboveClose), len(zArr)))
    dosSurf = np.append(patchBelow, dosSurf, axis = 0)
    dosSurf = np.append(dosSurf, patchAbove, axis = 0)
    return dosSurf

