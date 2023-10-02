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
    #wLO = 32.04 * 1e12
    #wTO = 7.92 * 1e12
    #epsInf = 6.3
    L = 40.
    zArr = np.logspace(-3, -9, 50, endpoint=True, base = 10)
    #zArr = np.append([L / 4.], zArr)

    #zArr = np.logspace(-3, -9, 100)

    #intNum = performSPhPIntegralNum(1. * 1e-8, wLO, wTO, epsInf)
    #print("intNum = {}".format(intNum[0] * 1e-20))
    #print("intNum Err = {}".format(intNum[1] * 1e-20))
#
    #dosAna, dosNum = computeSPhPIntAsOfZ(zArr, wLO, wTO, epsInf)
    #plotFreq.compareSPhPInt(dosAna, dosNum, zArr, "SPhPFieldA")

    #prod.produceFreqIntegralData(zArr, wLO, wTO, epsInf, L)
    producePlotAsOfFreq(zArr, wLO, wTO, epsInf, L)
    #computeFreqIntegralAsOfCutoff(zArr, wLO, wTO, epsInf, L)
    cutoff = 241.8 * 1e12# 1eV cutoff
    computeFreqIntegralFixedCutoff(zArr, cutoff, wLO, wTO, epsInf, L)
    #computeEffectiveHopping(zArr, cutoff, wLO, wTO, epsInf, L)


def computeFreqIntegralAsOfCutoff(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMTotal = prod.retrieveDosTM(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    #dosSurf = prod.retrieveDosSurf()



    dosIntTE = np.zeros(dosTETotal.shape)
    dosIntTM = np.zeros(dosTMTotal.shape)
    latConst = 1e-10

    for zInd, zVal in enumerate(zArr):
        for wInd, wVal in enumerate(wArr):
            wArrPart = wArr[:wInd]# * 1e-12
            #prefac = 2. * consts.fine_structure / np.pi * latConst ** 2 / consts.c ** 2 * wArrPart
            prefacFieldStrength = consts.hbar * wArrPart**3 / (2 * np.pi**2 * consts.epsilon_0 * consts.c**3)# * 1e24
            intFuncTE = prefacFieldStrength * (dosTETotal[:wInd, zInd] - .5)
            dosIntTE[wInd, zInd] = np.trapz(intFuncTE, x=wArrPart, axis = 0)
            intFuncTM = prefacFieldStrength * (dosTMTotal[:wInd, zInd] - .5)
            dosIntTM[wInd, zInd] = np.trapz(intFuncTM, x=wArrPart, axis = 0)

    #dosIntTM = dosIntTM + dosIntSurf

    filename = "TEField"
    plotFreq.plotDosIntegratedAsOfCutoff(dosIntTE, zArr, L, wArr, wLO, wTO, epsInf, filename)
    filename = "TMField"
    plotFreq.plotDosIntegratedAsOfCutoff(dosIntTM, zArr, L, wArr, wLO, wTO, epsInf, filename)

def computeFreqIntegralFixedCutoff(zArr, cutoff, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMTotal = prod.retrieveDosTM(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)

    dosIntTE = np.zeros(zArr.shape)
    dosIntTM = np.zeros(zArr.shape)

    for zInd, zVal in enumerate(zArr):
        #prefac = 2. * consts.fine_structure / np.pi * latConst ** 2 / consts.c ** 2 * wArrPart
        prefacE = consts.hbar * wArr**3 / (2 * np.pi**2 * consts.epsilon_0 * consts.c**3)
        prefacA = consts.hbar * wArr**1 / (2 * np.pi**2 * consts.epsilon_0 * consts.c**3) * 1e24
        prefacMass = 8. / (3. * np.pi) * consts.hbar / (consts.c**2 * consts.m_e)
        cutoffFac = np.exp(- wArr**2 / cutoff**2)
        intFuncTE = prefacMass * (dosTETotal[ : , zInd] - .5)# * cutoffFac
        dosIntTE[zInd] = np.trapz(intFuncTE, x=wArr, axis = 0)
        intFuncTM = prefacMass * (dosTMTotal[ : , zInd] - .5)# * cutoffFac
        dosIntTM[zInd] = np.trapz(intFuncTM, x=wArr, axis = 0)

    dosSurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        dosSurf[zInd] = performSPhPIntegralNum(zVal, wLO, wTO, epsInf)[0]

    dosTot = dosIntTE + dosIntTM + dosSurf
    dosNoSurf = dosIntTE# + dosIntTM

    filename = "EffectiveMass"
    #plotFreq.plotDosIntegratedFixedCutoff(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf, filename)
    plotFreq.plotEffectiveMass(dosTot, zArr, L, wLO, wTO, epsInf, filename)
    #filename = "TMEFieldCutoff"
    #plotFreq.plotDosIntegratedAsOfCutoff(dosIntTM, zArr, L, wArr, wLO, wTO, epsInf, filename)

def computeEffectiveHopping(zArr, cutoff, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMTotal = prod.retrieveDosTM(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)

    dosIntTE = np.zeros(zArr.shape)
    dosIntTM = np.zeros(zArr.shape)

    aLat = 1e-10

    for zInd, zVal in enumerate(zArr):
        hopFac = 0.5 * 2. * consts.fine_structure * aLat ** 2 / (3. * np.pi * consts.c ** 2) * wArr
        cutoffFac = np.exp(- wArr**2 / cutoff**2)
        intFuncTE = hopFac * (dosTETotal[ : , zInd] - .5) * cutoffFac
        dosIntTE[zInd] = np.trapz(intFuncTE, x=wArr, axis = 0)
        intFuncTM = hopFac * (dosTMTotal[ : , zInd] - .5) * cutoffFac
        dosIntTM[zInd] = np.trapz(intFuncTM, x=wArr, axis = 0)

    dosSurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        dosSurf[zInd] = performSPhPIntegralHopping(zVal, wLO, wTO, epsInf)[0]

    dosTot = dosIntTE + dosIntTM + dosSurf
    dosNoSurf = dosIntTE + dosIntTM

    filename = "HoppingCutoff"
    plotFreq.plotDosIntegratedHopping(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf, filename)
    #filename = "TMEFieldCutoff"
    #plotFreq.plotDosIntegratedAsOfCutoff(dosIntTM, zArr, L, wArr, wLO, wTO, epsInf, filename)


def computeLocalFieldStrength(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMTotal = prod.retrieveDosTM(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
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
    patchAbove = np.ones((len(arrAbove), len(zArr)))
    for wInd in range(len(arrAbove)):
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

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMTotal = prod.retrieveDosTM(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)

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
    prefacField = consts.hbar * omega**1 / (2. * consts.epsilon_0 * np.pi**2 * consts.c**3) * 1e24
    return prefacField * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncMass(omega, zVal, wLO, wTO, epsInf):
    prefacMass = 8. / (3. * np.pi) * consts.hbar / (consts.m_e * consts.c**2)
    return prefacMass * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncHopping(omega, zVal, wLO, wTO, epsInf):
    aLat = 1e-10
    hopFac = 0.5 * 2. * consts.fine_structure * aLat ** 2 / (3. * np.pi * consts.c ** 2) * omega
    return hopFac * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def performSPhPIntegralNum(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    #return scipy.integrate.quad(intFuncFieldStrength, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)
    return scipy.integrate.quad(intFuncMass, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)
    #return scipy.integrate.quad(intFuncEffectiveHopping, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 500)

def performSPhPIntegralHopping(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    return scipy.integrate.quad(intFuncHopping, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)

def performSPhPIntegralAna(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2)
    prefacField = consts.hbar * wInf**1 / (2. * consts.epsilon_0 * np.pi**2 * consts.c**3) * 1e24
    rho0Prefac = np.pi ** 2 * consts.c **3 / wInf**2
    rho = 1. / (8. * np.pi) * (wLO**2 - wTO**2) / wLO**2 / zVal**3
    return prefacField * rho0Prefac * rho
    #prefac = 2. * np.pi * consts.fine_structure * consts.c
    #num = epsInf* np.sqrt(1 + epsInf) * (wLO**2 - wTO**2)
    #denom = 4. * np.pi * np.sqrt(epsInf * wLO**2 + wTO**2) * (2. * epsInf * wLO**2 + (1 + epsInf**2) * wTO**2) * zVal**3
    #return prefac * num / denom

def computeSPhPIntAsOfZ(zArr, wLO, wTO, epsInf):
    intAna = performSPhPIntegralAna(zArr, wLO, wTO, epsInf)
    intNum = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        intNum[zInd] = performSPhPIntegralNum(zVal, wLO, wTO, epsInf)[0]

    return (intAna, intNum)

def patchDosSurfWithZeros(dosSurf, zArr, arrBelow, arrAboveClose):
    patchBelow = np.zeros((len(arrBelow), len(zArr)))
    patchAbove = np.zeros((len(arrAboveClose), len(zArr)))
    dosSurf = np.append(patchBelow, dosSurf, axis = 0)
    dosSurf = np.append(dosSurf, patchAbove, axis = 0)
    return dosSurf

