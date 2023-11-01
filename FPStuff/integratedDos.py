import numpy as np
import scipy.constants as consts
import scipy.integrate
import math

import plotDosFP
import computeDosFP

def dosParallelOmegaWithReg(omega, zVal, L, alpha):
    return (computeDosFP.dosParallelOneFreq(omega, zVal, L) - 2. / 3.) * (omega * 1e-12) * np.exp(- alpha**2 * omega**2)

def dosPerpOmegaWithReg(omega, zVal, L, alpha):
    return (computeDosFP.dosPerpOneFreq(omega, zVal, L) - 1. / 3.) * (omega * 1e-12) * np.exp(- alpha**2 * omega**2)

def dosTEWithReg(omega, zVal, L, alpha):
    return (computeDosFP.dosTEOneFreq(omega, zVal, L) - 1. / 2.) * (omega * 1e-12) * np.exp(- alpha**2 * omega**2)

def dosTMWithReg(omega, zVal, L, alpha):
    return (computeDosFP.dosTMOneFreq(omega, zVal, L) - 1. / 2.) * (omega * 1e-12) * np.exp(- alpha**2 * omega**2)

def dosParallelWithRegE(omega, zVal, L, alpha):
    return (computeDosFP.dosParallelOneFreq(omega, zVal, L) - 2. / 3.) * (omega * 1e-12)**3 * np.exp(- alpha**2 * omega**2)

def dosParallelWithRegA(omega, zVal, L, alpha):
    return (computeDosFP.dosParallelOneFreq(omega, zVal, L) - 2. / 3.) * (omega * 1e-12)**1 * np.exp(- alpha**2 * omega**2)

def dosEffectiveMassReg(omega, zVal, L, alpha):
    return (computeDosFP.dosParallelOneFreq(omega, zVal, L) - 2. / 3.) * np.exp(- alpha**2 * omega**2)

def dosPerpWithRegE(omega, zVal, L, alpha):
    return (computeDosFP.dosPerpOneFreq(omega, zVal, L) - 1. / 3.) * (omega * 1e-12)**3 * np.exp(- alpha**2 * omega**2)

def dosTEWithRegE(omega, zVal, L, alpha):
    return (computeDosFP.dosTEOneFreq(omega, zVal, L) - 1. / 2.) * (omega * 1e-12)**3 * np.exp(- alpha**2 * omega**2)

def dosTMWithRegE(omega, zVal, L, alpha):
    return (computeDosFP.dosTMOneFreq(omega, zVal, L) - 1. / 2.) * (omega * 1e-12)**3 * np.exp(- alpha**2 * omega**2)

def dosSumRuleWithReg(omega, zVal, L, alpha):
    #return (computeDosFP.dosTEOneFreq(omega, zVal, L) + 0. * computeDosFP.dosTMOneFreq(omega, zVal, L) - .5) * np.exp(- alpha**2 * omega**2)
    return (computeDosFP.dosTEOneFreq(omega, zVal, L) - .5) * np.exp(- alpha**2 * omega**2)

def numericalIntegral(zVal, omegaMax, L):
    cutoffArr = np.logspace(11, np.log10(omegaMax), 30)
    alphaArr = 1. / cutoffArr * 22
    resArr = np.zeros(len(cutoffArr))
    for cutInd, cutoff in enumerate(cutoffArr):
        print("cutoff = {}THz".format(cutoff * 1e-12))
        numResonances = math.floor(cutoff * L / np.pi / consts.c)
        resonancePoints = (np.arange(numResonances) + 1.) * np.pi * consts.c / L
        res = scipy.integrate.quad(dosParallelWithRegE, 0, cutoff, args=(zVal, L, alphaArr[cutInd]), points=resonancePoints, limit = (numResonances * 10 + 50))
        resArr[cutInd] = res[0]

    fieldFac = consts.hbar / (2. * consts.epsilon_0 * np.pi ** 2 * consts.c ** 3) * 1e36
    #plotDosFP.plotFieldIntegrals(cutoffArr, resArr * fieldFac)
    return (cutoffArr, resArr * fieldFac)

def numericalIntegralEField(cutoff, dArr):
    wMax = 10. * cutoff
    integratedDos = np.zeros(len(dArr))
    for dInd, dVal in enumerate(dArr):
        print("d = {}m".format(dVal))
        numResonances = math.floor(wMax * dVal / np.pi / consts.c)
        resonancePoints = (np.arange(numResonances) + 1.) * np.pi * consts.c / dVal
        res = scipy.integrate.quad(dosParallelWithRegE, 0, wMax, args=(dVal / 2., dVal, 1. / cutoff), points=resonancePoints, limit = (numResonances * 4 + 50))
        #res = scipy.integrate.quad(dosEffectiveMassReg, 0, wMax, args=(dVal / 2., dVal, 1. / cutoff), points=resonancePoints, limit = (numResonances * 4 + 50))
        integratedDos[dInd] = res[0]

    fieldFac = consts.hbar / (2. * consts.epsilon_0 * np.pi ** 2 * consts.c ** 3) * 1e36
    #massFac = 8. / (3. * np.pi) * consts.hbar / (consts.m_e * consts.c ** 2)
    return fieldFac * integratedDos

def numericalIntegralAField(cutoff, dArr):
    wMax = 4. * cutoff
    integratedDos = np.zeros(len(dArr))
    for dInd, dVal in enumerate(dArr):
        print("d = {}m".format(dVal))
        numResonances = math.floor(wMax * dVal / np.pi / consts.c)
        resonancePoints = (np.arange(numResonances) + 1.) * np.pi * consts.c / dVal
        res = scipy.integrate.quad(dosParallelWithRegA, 0, wMax, args=(dVal / 2., dVal, 1. / cutoff), points=resonancePoints, limit = (numResonances * 4 + 50))
        integratedDos[dInd] = res[0]

    fieldFac = consts.hbar / (2. * consts.epsilon_0 * np.pi ** 2 * consts.c ** 3) * 1e36
    return fieldFac * integratedDos

def numericalIntegralSumRule(cutoff, dArr):
    wMax = 10. * cutoff
    integratedDos = np.zeros(len(dArr))
    for dInd, dVal in enumerate(dArr):
        print("d = {}m".format(dVal))
        numResonances = math.floor(wMax * dVal / np.pi / consts.c)
        resonancePoints = (np.arange(numResonances) + 1.) * np.pi * consts.c / dVal
        resonancePoints = np.append([0], resonancePoints)
        res = scipy.integrate.quad(dosSumRuleWithReg, 0, wMax, args=(dVal / 2., dVal, 1. / cutoff), points=resonancePoints, limit = (numResonances * 4 + 50))
        integratedDos[dInd] = res[0]

    return integratedDos

def numericalIntegralEffectiveMass(cutoff, dArr):
    wMax = 4. * cutoff
    massArr = np.zeros(len(dArr))
    for dInd, dVal in enumerate(dArr):
        print("d = {}m".format(dVal))
        numResonances = math.floor(wMax * dVal / np.pi / consts.c)
        resonancePoints = (np.arange(numResonances) + 1.) * np.pi * consts.c / dVal
        res = scipy.integrate.quad(dosEffectiveMassReg, 0, wMax, args=(dVal / 2., dVal, 1. / cutoff), points=resonancePoints, limit = (numResonances * 4 + 50))
        massArr[dInd] = res[0]
    prefacMass = 16. / (3. * np.pi) * consts.hbar / (consts.c**2 * consts.m_e)

    return prefacMass * massArr

def numericalIntegralHopping(cutoff, dArr):
    wMax = 4. * cutoff
    fieldArr = np.zeros(len(dArr))
    for dInd, dVal in enumerate(dArr):
        print("d = {}m".format(dVal))
        numResonances = math.floor(wMax * dVal / np.pi / consts.c)
        resonancePoints = (np.arange(numResonances) + 1.) * np.pi * consts.c / dVal
        res = scipy.integrate.quad(dosParallelOmegaWithReg, 1e9, wMax, args=(dVal / 2., dVal, 1. / cutoff), points=resonancePoints, limit = (numResonances * 4 + 50))
        #print("Int res = {}".format(res))
        fieldArr[dInd] = res[0]
    aLat = 1e-10
    #factor 0.5 for having only (say) x-direction
    hopFac = 0.5 * 2. * consts.fine_structure * aLat**2 / (3. * np.pi * consts.c**2) * 1e12
    return hopFac * fieldArr

def computeIntegral(omArr, dos, L):
    cutoffArr = np.logspace(11, np.log10(np.amax(omArr)), 20)
    alphaArr = 1. / cutoffArr * 5.
    dosIntegrand = (dos - 2. / 3.) * omArr**1
    resArr = np.zeros(len(alphaArr))
    for ind, alpha in enumerate(alphaArr):
        partialOmInds = np.where(omArr < cutoffArr[ind])
        resArr[ind] = integrateDosWithReg(omArr[partialOmInds], dosIntegrand[partialOmInds], alpha)
    fieldFac = consts.hbar / (2. * consts.epsilon_0 * np.pi**2 * consts.c**3) * 1e24
    plotDosFP.plotFieldIntegrals(cutoffArr, resArr * fieldFac)
    return resArr

def integrateDosWithReg(omArr, dos, convFac):
    dos = dos * np.exp(-convFac * omArr)
    #plotDosFP.plotDosWithCutoff(omArr, dos)
    integral = np.trapz(dos, x = omArr)
    return integral




