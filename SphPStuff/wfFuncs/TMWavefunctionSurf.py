import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import h5py
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import complexIntegral
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.constants as consts

import SphPStuff.findAllowedKsSPhP as findAllowedKsSPhP
import SphPStuff.epsilonFunctions as epsFunc


fontsize = 8

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 8
mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble'] = [
    #    r'\renewcommand{\familydefault}{\sfdefault}',
    #    r'\usepackage[scaled=1]{helvet}',
    r'\usepackage[helvet]{sfmath}',
    #    r'\everymath={\sf}'
]

def NormSqr(kVal, L, omega, wLO, wTO, epsInf):
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    brack11 = np.exp(- kVal * L) * omega**2 / (consts.c**2 * kVal**2)
    brack12 = (omega**2 / (consts.c**2 * kVal**2) + 2) * (1 - np.exp(- 2 * kVal * L)) / (2 * kVal * L)
    term1 = (brack11 + brack12) * 0.25 * (1 + np.exp(-2. * kDVal * L) - 2. * np.exp(-kDVal * L))

    brack21 = np.exp(- kDVal * L) * eps * omega ** 2 / (consts.c ** 2 * kDVal ** 2)
    brack22 = (eps * omega ** 2 / (consts.c ** 2 * kDVal ** 2) + 2) * (1 - np.exp(- 2 * kDVal * L)) / (2 * kDVal * L)
    term2 = (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kVal * L) - 2. * np.exp(-kVal * L))

    hopfieldFactor = 1 + wTO**2 * (wLO**2 - wTO**2) / (wTO**2 - omega**2)**2

    return L / 4. * (term1 + term2) * hopfieldFactor

def waveFunctionPosPara(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    func = 0.25 * (np.exp(- kVal * zArr) - np.exp(kVal * (zArr - L))) * (1 - np.exp(-kDVal * L))
    return 1. / np.sqrt(NSqr) * func

def waveFunctionNegPara(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    func = 0.25 * (np.exp(kDVal * zArr) - np.exp(- kDVal * (L + zArr))) * (1 - np.exp(-kVal * L))
    return 1. / np.sqrt(NSqr) * func

def waveFunctionPosPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    func = 0.25 * np.sqrt(omega**2 / (consts.c**2 * kVal**2) + 1) * ( np.exp(- kVal * zArr) +  np.exp(kVal * (zArr - L))) * (1 - np.exp(-kDVal * L))
    return 1. / np.sqrt(NSqr) * func

def waveFunctionNegPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    func = -0.25 * np.sqrt(eps * omega**2 / (consts.c**2 * kDVal**2) + 1) * ( np.exp(kDVal * zArr) +  np.exp(-kDVal * (L + zArr))) * (1 - np.exp(-kVal * L))
    return 1. / np.sqrt(NSqr) * func



def waveFunctionTMPara(zArr, kArr, L, omega, wLO, wTO, epsInf):
    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPosPara(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
    wFNeg = waveFunctionNegPara(zNegArr, kArr, L, omega, wLO, wTO, epsInf)

    wF = np.append(wFNeg, wFPos)

    return wF


def waveFunctionTMPerp(zArr, kArr, L, omega, wLO, wTO, epsInf):
    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPosPerp(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
    wFNeg = waveFunctionNegPerp(zNegArr, kArr, L, omega, wLO, wTO, epsInf)

    wF = np.append(wFNeg, wFPos)

    return wF

def NSqAna(kVal, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    kpara = np.sqrt(omega**2 / consts.c**2 + kVal**2)
    hopfieldFactor = 1 + wTO**2 * (wLO**2 - wTO**2) / (wTO**2 - omega**2)**2
    return hopfieldFactor / (2. * kpara * np.sqrt(epsAbs)) * (1 + epsAbs) * (epsAbs + 1. / epsAbs)

def waveFunctionPosParaAna(zArr, kVal, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kpara = np.sqrt(omega**2 / consts.c**2 + kVal**2)
    NSq = NSqAna(kVal, omega, wLO, wTO, epsInf)
    return 1. / np.sqrt(NSq) * np.exp(- np.sqrt(1. / np.abs(eps)) * kpara * zArr)

def waveFunctionNegParaAna(zArr, kVal, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kpara = np.sqrt(omega**2 / consts.c**2 + kVal**2)
    NSq = NSqAna(kVal, omega, wLO, wTO, epsInf)
    return 1./np.sqrt(NSq) * np.exp(np.sqrt(np.abs(eps)) * kpara * zArr)

def waveFunctionTMParaAna(zArr, kArr, omega, wLO, wTO, epsInf):
    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPosParaAna(zPosArr, kArr, omega, wLO, wTO, epsInf)
    wFNeg = waveFunctionNegParaAna(zNegArr, kArr, omega, wLO, wTO, epsInf)

    wF = np.append(wFNeg, wFPos)

    return wF

def waveFunctionPosPerpAna(zArr, kVal, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kpara = np.sqrt(omega**2 / consts.c**2 + kVal**2)
    NSq = NSqAna(kVal, omega, wLO, wTO, epsInf)
    return np.sqrt(np.abs(eps)) / np.sqrt(NSq) * np.exp(- np.sqrt(1. / np.abs(eps)) * kpara * zArr)

def waveFunctionNegPerpAna(zArr, kVal, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kpara = np.sqrt(omega**2 / consts.c**2 + kVal**2)
    NSq = NSqAna(kVal, omega, wLO, wTO, epsInf)
    return - 1. / (np.sqrt(NSq) * np.sqrt(np.abs(eps))) * np.exp(np.sqrt(np.abs(eps)) * kpara * zArr)

def waveFunctionTMPerpAna(zArr, kArr, omega, wLO, wTO, epsInf):
    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPosPerpAna(zPosArr, kArr, omega, wLO, wTO, epsInf)
    wFNeg = waveFunctionNegPerpAna(zNegArr, kArr, omega, wLO, wTO, epsInf)

    wF = np.append(wFNeg, wFPos)

    return wF


def waveFunctionForInt(z, k, L, omega, wLO, wTO, epsInf):
    if (z >= 0):
        return (waveFunctionPosPara(z, k, L, omega, wLO, wTO, epsInf)) ** 2 + (waveFunctionPosPerp(z, k, L, omega, wLO, wTO, epsInf)) ** 2
    else:
        return (waveFunctionNegPara(z, k, L, omega, wLO, wTO, epsInf)) ** 2 + (waveFunctionNegPerp(z, k, L, omega, wLO, wTO, epsInf)) ** 2

def checkNormalizationK(k, L, omega, wLO, wTO, epsInf):
    checkInt = integrate.quad(waveFunctionForInt, -L / 200., L / 200., args=(k, L, omega, wLO, wTO, epsInf))
    print("Norm = {}, with Int accuracy = {}".format(checkInt[0], checkInt[1]))

def checkNormalizations(allowedKs, L, omega, wLO, wTO, epsInf):
    for kVal in allowedKs:
        checkNormalizationK(kVal, L, omega, wLO, wTO, epsInf)

def waveFunctionForIntAna(z, k, omega, wLO, wTO, epsInf):
    if (z >= 0):
        return (waveFunctionPosParaAna(z, k, omega, wLO, wTO, epsInf)) ** 2 + (waveFunctionPosPerpAna(z, k, omega, wLO, wTO, epsInf)) ** 2
    else:
        return (waveFunctionNegParaAna(z, k, omega, wLO, wTO, epsInf)) ** 2 + (waveFunctionNegPerpAna(z, k, omega, wLO, wTO, epsInf)) ** 2

def checkNormalizationKAna(k, L, omega, wLO, wTO, epsInf):
    checkInt = integrate.quad(waveFunctionForIntAna, -L / 200., L / 200., args=(k, omega, wLO, wTO, epsInf))
    print("Norm = {}, with Int accuracy = {}".format(checkInt[0], checkInt[1]))

def checkNormalizationsAna(allowedKs, L, omega, wLO, wTO, epsInf):
    for kVal in allowedKs:
        checkNormalizationKAna(kVal, L, omega, wLO, wTO, epsInf)


def plotWaveFunctionPara(kArr, zArr, L, omega, wLO, wTO, epsInf):
    wF = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    wFAna = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kValAna = omega / consts.c * 1. / np.sqrt(np.abs(eps) - 1)

    for kInd, kVal in enumerate(kArr):
        wF[kInd, :] = waveFunctionTMPara(zArr, kVal, L, omega, wLO, wTO, epsInf)
        wFAna[kInd, :] = waveFunctionTMParaAna(zArr, kValAna, omega, wLO, wTO, epsInf)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    for kInd, kVal in enumerate(kArr):
        ax.plot(zArr, wF[kInd, :], color=cmapPink(.3), lw=1.)
        ax.plot(zArr, wFAna[kInd, :], color=cmapPink(.6), lw=1., linestyle = '--')

    ax.axhline(0, lw=0.5, color='gray')
    ax.axvline(0, lw=0.5, color='gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    #ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    #ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./SPhPPlotsSaved/wFSPhPTMSurfPara.png")

def plotWaveFunctionPerp(kDArr, zArr, L, omega, wLO, wTO, epsInf):
    wF = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)
    wFAna = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)

    for kDInd, kDVal in enumerate(kDArr):
        wF[kDInd, :] = waveFunctionTMPerp(zArr, kDVal, L, omega, wLO, wTO, epsInf)
        wFAna[kDInd, :] = waveFunctionTMPerpAna(zArr, kDVal, omega, wLO, wTO, epsInf)

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    indNeg = np.where(zArr < 0)
    wF[:, indNeg] = wF[:, indNeg] * eps
    wFAna[:, indNeg] = wFAna[:, indNeg] * eps

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapBone = cm.get_cmap('bone')
    for kDInd, kDVal in enumerate(kDArr):
        ax.plot(zArr, wF[kDInd, :], color=cmapBone(0.3), lw=1.)
        ax.plot(zArr, wFAna[kDInd, :], color=cmapBone(0.6), lw=1., linestyle = '--')

    ax.axhline(0, lw=0.5, color='gray')
    ax.axvline(0, lw=0.5, color='gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{200}$", r"$0$", r"$\frac{L}{200}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$ \varepsilon(\omega) f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./SPhPPlotsSaved/wFSPhPTMSurfPerp.png")


def createPlotSurf():
    epsInf = 2.
    omega = 2. * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    L = 1.

    #zArr = np.linspace(-consts.c / omega * 5., consts.c / omega * 5., 1000)
    zArr = np.linspace(- L / 200., L / 200., 1000)

    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "Surf")
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kValAna = omega / consts.c * 1. / np.sqrt((np.abs(eps) - 1))
    print("kVal numerical = {}".format(allowedKs[0]))
    print("kVal analytical = {}".format(kValAna))



    checkNormalizations(allowedKs, L, omega, wLO, wTO, epsInf)
    checkNormalizationsAna(allowedKs, L, omega, wLO, wTO, epsInf)
    plotWaveFunctionPara(allowedKs, zArr, L, omega, wLO, wTO, epsInf)
    plotWaveFunctionPerp(allowedKs, zArr, L, omega, wLO, wTO, epsInf)
