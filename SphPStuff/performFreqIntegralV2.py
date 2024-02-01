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
import produceFreqDataV2 as prodV2
import fluctuationPlotsThesis as plotFlucThesis

def freqIntegral(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):

    evCutoff = 1519.3 * 1e12 # 1eV
    #cutoff = .1 * evCutoff
    #For scaling plot
    cutoff = .075 * evCutoff
    computeEffectiveMass(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L)
    #computeEffectiveHopping(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L)
    #computeFluctuations(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L)
    #computeFluctuationsMultipleCutoffs(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

def computeFreqIntegralAsOfCutoff(wArr, zArr, wLO, wTO, epsInf, L):
    dosTETotal = prodV2.retrieveDosTE(wArr, L, wLO, wTO, epsInf)
    dosTMTotal = prodV2.retrieveDosTMTotal(wArr, L, wLO, wTO, epsInf)
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

def computeEffectiveMass(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L):

    dosTETotal, dosTMPara = prodV2.retrieveDosPara(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)
    wArr = prodV2.defineFreqArrayOne(wArrSubdivisions)

    dosIntTE = np.zeros(zArr.shape)
    dosIntTM = np.zeros(zArr.shape)

    for zInd, zVal in enumerate(zArr):
        prefacMass = 16. / (3. * np.pi) * consts.hbar / (consts.c**2 * consts.m_e)
        cutoffFac = np.exp(- wArr**2 / cutoff**2)
        intFuncTE = prefacMass * (dosTETotal[ : , zInd] - .5)# * cutoffFac
        intFuncTE[:9] = 0.
        dosIntTE[zInd] = np.trapz(intFuncTE, x=wArr, axis = 0)
        intFuncTM = prefacMass * (dosTMPara[ : , zInd] - 1. / 6.)# * cutoffFac
        intFuncTM[:9] = 0.
        dosIntTM[zInd] = np.trapz(intFuncTM, x=wArr, axis = 0)

    dosSurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        dosSurf[zInd] = performSPhPIntegralMass(zVal, wLO, wTO, epsInf)[0]

    dosTot = dosIntTE + dosIntTM + dosSurf
    dosBulk = dosIntTE + dosIntTM

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filename = "./savedData/massForPaper" + wLOStr + wTOStr + ".hdf5"
    print("Writing masses to file: " + filename)
    prod.writeMasses(cutoff, zArr, dosTot, dosBulk, filename)

    #filename = "EffectiveMass"
    #plotFreq.plotEffectiveMass(dosTot, zArr, L, wLO, wTO, epsInf, filename)


def produceCollapsePlotE(wArr, zArr, cutoff, wLOArr, wTOArr, epsInf, L):

    flucEArr = np.zeros((2, len(wLOArr), len(zArr)))
    for wInd, _ in enumerate(wLOArr):
        cutoff = 3. * wLOArr[wInd]
        flucEArr[:, wInd, :] = computeFluctuationsE(wArr, zArr, cutoff, wLOArr[wInd], wTOArr[wInd], epsInf, L)

    plotFreq.collapseFlucE(flucEArr[0, :, :], flucEArr[1, :, :], zArr, L, wLOArr, wTOArr, epsInf)

def produceCollapsePlotA(wArr, zArr, cutoff, wLOArr, wTOArr, epsInf, L):

    flucAArr = np.zeros((2, len(wLOArr), len(zArr)))
    for wInd, _ in enumerate(wLOArr):
        cutoff = 3. * wLOArr[wInd]
        flucAArr[:, wInd, :] = computeFluctuationsA(wArr, zArr, cutoff, wLOArr[wInd], wTOArr[wInd], epsInf, L)

    plotFreq.collapseFlucA(flucAArr[0, :, :], flucAArr[1, :, :], zArr, L, wLOArr, wTOArr, epsInf)


def computeFluctuationsE(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L):

    dosTETotal, dosTMTotal = prodV2.retrieveDosPara(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

    wArr = prodV2.defineFreqArrayOne(wArrSubdivisions)

    flucETE = np.zeros(zArr.shape)
    flucETM = np.zeros(zArr.shape)

    for zInd, zVal in enumerate(zArr):
        cutoffFac = np.exp(- wArr**2 / cutoff**2)
        prefacE = consts.hbar * wArr**3 / (2 * np.pi**2 * consts.epsilon_0 * consts.c**3)
        flucETEInt = prefacE * (dosTETotal[ : , zInd] - .5) * cutoffFac
        flucETE[zInd] = np.trapz(flucETEInt, x=wArr, axis = 0)
        flucETMInt = prefacE * (dosTMTotal[ : , zInd] - 1. / 6.) * cutoffFac
        flucETM[zInd] = np.trapz(flucETMInt, x=wArr, axis = 0)

    flucESurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        flucESurf[zInd] = performSPhPIntegralEFluc(zVal, wLO, wTO, epsInf)[0]

    flucE = flucETE + flucETM + flucESurf
    flucENoSurf = flucETE + flucETM

    return flucENoSurf, flucE


def computeFluctuationsA(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L):

    dosTETotal, dosTMTotal = prodV2.retrieveDosPara(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)
    wArr = prodV2.defineFreqArrayOne(wArrSubdivisions)

    flucATE = np.zeros(zArr.shape)
    flucATM = np.zeros(zArr.shape)

    for zInd, zVal in enumerate(zArr):
        cutoffFac = np.exp(- wArr**2 / cutoff**2)
        prefacA = consts.hbar * wArr**1 / (2 * np.pi**2 * consts.epsilon_0 * consts.c**3)
        flucATEInt = prefacA * (dosTETotal[ : , zInd] - .5) * cutoffFac
        flucATE[zInd] = np.trapz(flucATEInt, x=wArr, axis = 0)
        flucATMInt = prefacA * (dosTMTotal[ : , zInd] - 1. / 6.) * cutoffFac
        flucATM[zInd] = np.trapz(flucATMInt, x=wArr, axis = 0)

    flucASurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        flucASurf[zInd] = performSPhPIntegralAFluc(zVal, wLO, wTO, epsInf)[0]

    flucA = flucATE + flucATM + flucASurf
    flucANoSurf = flucATE + flucATM

    return flucANoSurf, flucA


def computeFluctuations(wArrSubdivision, zArr, cutoff, wLO, wTO, epsInf, L):

    flucENoSurf, flucE = computeFluctuationsE(wArrSubdivision, zArr, cutoff, wLO, wTO, epsInf, L)
    flucANoSurf, flucA = computeFluctuationsA(wArrSubdivision, zArr, cutoff, wLO, wTO, epsInf, L)

    #plotFreq.plotFluctuationsEandA(flucENoSurf, flucE, flucANoSurf, flucA, zArr, L, wLO, wTO, epsInf)
    #plotFreq.plotFluctuationsEandANaturalUnits(flucENoSurf, flucE, flucANoSurf, flucA, zArr, L, wLO, wTO, epsInf)
    #plotFreq.plotFluctuationsENaturalUnits(flucENoSurf, flucE, zArr, L, wLO, wTO, epsInf)
    #plotFreq.plotFluctuationsANaturalUnits(flucANoSurf, flucA, zArr, L, wLO, wTO, epsInf)

    plotFlucThesis.plotFluctuationsENaturalUnits(flucENoSurf, flucE, zArr, L, wLO, wTO, epsInf)
    plotFlucThesis.plotFluctuationsANaturalUnits(flucANoSurf, flucA, zArr, L, wLO, wTO, epsInf)

    #plotFlucThesis.plotFluctuationsEExpUnits(flucENoSurf, flucE, zArr, L, wLO, wTO, epsInf)

def computeFluctuationsMultipleCutoffs(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):
    evCutoff = 1519.3 * 1e12 #1eV
    cutoffArr = np.logspace(np.log10(0.02 * evCutoff), np.log10(.2 * evCutoff), 5, endpoint=True)
    print(cutoffArr * 1e-12)

    fluctuationsTotE = np.zeros((len(cutoffArr), len(zArr)), dtype = float)
    fluctuationsNoSurfE = np.zeros((len(cutoffArr), len(zArr)), dtype = float)
    fluctuationsTotA = np.zeros((len(cutoffArr), len(zArr)), dtype = float)
    fluctuationsNoSurfA = np.zeros((len(cutoffArr), len(zArr)), dtype = float)
    for cutoffInd, cutoff in enumerate(cutoffArr):
        fluctuationsNoSurfE[cutoffInd, :], fluctuationsTotE[cutoffInd, :]  = computeFluctuationsE(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L)
        fluctuationsNoSurfA[cutoffInd, :], fluctuationsTotA[cutoffInd, :]  = computeFluctuationsA(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L)

    plotFlucThesis.plotFluctuationsECutoffConv(fluctuationsNoSurfE[:, :], fluctuationsTotE[:, :], cutoffArr, zArr, L, wLO, wTO, epsInf)
    plotFlucThesis.plotFluctuationsACutoffConv(fluctuationsNoSurfA[:, :], fluctuationsTotA[:, :], cutoffArr, zArr, L, wLO, wTO, epsInf)


def computeEffectiveHopping(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L):


    dosTETotal, dosTMTotal = prodV2.retrieveDosPara(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)
    wArr = prodV2.defineFreqArrayOne(wArrSubdivisions)

    dosIntTE = np.zeros(zArr.shape)
    dosIntTM = np.zeros(zArr.shape)

    aLat = 1e-10

    for zInd, zVal in enumerate(zArr):
        hopFac = 0.5 * 2. * consts.fine_structure * aLat ** 2 / (3. * np.pi * consts.c ** 2) * wArr
        cutoffFac = np.exp(- wArr**2 / cutoff**2)
        intFuncTE = hopFac * (dosTETotal[ : , zInd] - .5) * cutoffFac
        dosIntTE[zInd] = np.trapz(intFuncTE, x=wArr, axis = 0)
        intFuncTM = hopFac * (dosTMTotal[ : , zInd] - 1. / 6.) * cutoffFac
        dosIntTM[zInd] = np.trapz(intFuncTM, x=wArr, axis = 0)

    dosSurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        dosSurf[zInd] = performSPhPIntegralHopping(zVal, wLO, wTO, epsInf)[0]

    dosTot = dosIntTE + dosIntTM + dosSurf

    prod.writeMasses(cutoff, zArr, dosTot, "savedData/hoppingThesis1ForPlottingThesis.hdf5")

    #filename = "HoppingCutoff"
    #plotFreq.plotDosIntegratedHopping(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf, filename)
    #filename = "TMEFieldCutoff"
    #plotFreq.plotDosIntegratedAsOfCutoff(dosIntTM, zArr, L, wArr, wLO, wTO, epsInf, filename)

def intFuncEffectiveHopping(omega, zVal, wLO, wTO, epsInf):
    prefac = 2. * consts.fine_structure / np.pi / consts.c ** 2 * omega # / 3.
    return prefac * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncFieldStrengthA(omega, zVal, wLO, wTO, epsInf):
    epsilon = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    prefacPara = 1. / (1. + np.abs(epsilon))
    prefacField = consts.hbar * omega**1 / (2. * consts.epsilon_0 * np.pi**2 * consts.c**3)
    return prefacPara * prefacField * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncFieldStrengthE(omega, zVal, wLO, wTO, epsInf):
    epsilon = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    prefacPara = 1. / (1. + np.abs(epsilon))
    prefacField = consts.hbar * omega**3 / (2. * consts.epsilon_0 * np.pi**2 * consts.c**3)
    return prefacPara * prefacField * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intSumRule(omega, zVal, wLO, wTO, epsInf):
    return dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncMass(omega, zVal, wLO, wTO, epsInf):
    epsilon = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    prefacPara = 1. / (1. + np.abs(epsilon))
    prefacMass = 16. / (3. * np.pi) * consts.hbar / (consts.m_e * consts.c**2)
    return prefacPara * prefacMass * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def intFuncHopping(omega, zVal, wLO, wTO, epsInf):
    epsilon = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    prefacPara = 1. / (1. + np.abs(epsilon))
    aLat = 1e-10
    hopFac = 0.5 * 2. * consts.fine_structure * aLat ** 2 / (3. * np.pi * consts.c ** 2) * omega
    return hopFac * prefacPara * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def performSPhPIntegralMass(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    #return scipy.integrate.quad(intFuncFieldStrength, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)
    return scipy.integrate.quad(intFuncMass, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 10000)
    #return scipy.integrate.quad(intFuncEffectiveHopping, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 500)

def performSPhPIntegralEFluc(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    return scipy.integrate.quad(intFuncFieldStrengthE, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)

def performSPhPIntegralAFluc(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    return scipy.integrate.quad(intFuncFieldStrengthA, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)

def performSPhPIntegralSumRule(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    return scipy.integrate.quad(intSumRule, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 1000)

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
        intNum[zInd] = performSPhPIntegralMass(zVal, wLO, wTO, epsInf)[0]

    return (intAna, intNum)

def patchDosSurfWithZeros(dosSurf, zArr, arrBelow, arrAboveClose):
    patchBelow = np.zeros((len(arrBelow), len(zArr)))
    patchAbove = np.zeros((len(arrAboveClose), len(zArr)))
    dosSurf = np.append(patchBelow, dosSurf, axis = 0)
    dosSurf = np.append(dosSurf, patchAbove, axis = 0)
    return dosSurf

