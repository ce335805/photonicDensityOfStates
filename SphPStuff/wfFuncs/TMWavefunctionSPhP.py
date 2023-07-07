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
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    brack11 = omega**2 / (consts.c**2 * kVal**2)
    brack12 = (omega**2 / (consts.c**2 * kVal**2) - 2) * np.sin(kVal * L) / (kVal * L)
    term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
    brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
    brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
    term2 = (brack21 + brack22) * np.sin(kVal * L / 2.)**2

    return L / 4. * (term1 + term2)

def waveFunctionPosPara(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    func = np.sin(kVal * (L / 2 - zArr)) * np.sin(kDVal * L / 2.)
    return 1. / np.sqrt(NSqr) * func

def waveFunctionNegPara(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    func = np.sin(kDVal * (L / 2. + zArr)) * np.sin(kVal * L / 2.)
    return 1. / np.sqrt(NSqr) * func

def waveFunctionPosPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    func = np.sqrt(omega**2 / (consts.c**2 * kVal**2) - 1) * np.cos(kVal * (L / 2 - zArr)) * np.sin(kDVal * L / 2.)
    return 1. / np.sqrt(NSqr) * func

def waveFunctionNegPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    func = - np.sqrt(eps * omega**2 / (consts.c**2 * kDVal**2) - 1) * np.cos(kDVal * (L / 2. + zArr)) * np.sin(kVal * L / 2.)
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

def waveFunctionForInt(z, k, L, omega, wLO, wTO, epsInf):
    if (z >= 0):
        return (waveFunctionPosPara(z, k, L, omega, wLO, wTO, epsInf)) ** 2 + (waveFunctionPosPerp(z, k, L, omega, wLO, wTO, epsInf)) ** 2
    else:
        return (waveFunctionNegPara(z, k, L, omega, wLO, wTO, epsInf)) ** 2 + (waveFunctionNegPerp(z, k, L, omega, wLO, wTO, epsInf)) ** 2

def checkNormalizationK(k, L, omega, wLO, wTO, epsInf):
    checkInt = integrate.quad(waveFunctionForInt, -L / 2., L / 2., args=(k, L, omega, wLO, wTO, epsInf))
    print("Norm = {}, with Int accuracy = {}".format(checkInt[0], checkInt[1]))

def checkNormalizations(allowedKs, L, omega, wLO, wTO, epsInf):
    for kVal in allowedKs[:10]:
        checkNormalizationK(kVal, L, omega, wLO, wTO, epsInf)

def plotWaveFunctionPara(kDArr, zArr, L, omega, wLO, wTO, epsInf):
    wF = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)

    for kDInd, kDVal in enumerate(kDArr):
        wF[kDInd, :] = waveFunctionTMPara(zArr, kDVal, L, omega, wLO, wTO, epsInf)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    for kDInd, kDVal in enumerate(kDArr):
        color = cmapPink(kDInd / (wF.shape[0] + 0.5))
        ax.plot(zArr, wF[kDInd, :], color=color, lw=1.)

    ax.axhline(0, lw=0.5, color='gray')
    ax.axvline(0, lw=0.5, color='gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{20}$", r"$0$", r"$\frac{L}{20}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./SPhPPlotsSaved/wFSPhPTMPara.png")

def plotWaveFunctionPerp(kDArr, zArr, L, omega, wLO, wTO, epsInf):
    wF = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)

    for kDInd, kDVal in enumerate(kDArr):
        wF[kDInd, :] = waveFunctionTMPerp(zArr, kDVal, L, omega, wLO, wTO, epsInf)

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    indNeg = np.where(zArr < 0)
    wF[:, indNeg] = wF[:, indNeg] * eps

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    for kDInd, kDVal in enumerate(kDArr):
        color = cmapPink(kDInd / (wF.shape[0] + 0.5))
        ax.plot(zArr, wF[kDInd, :], color=color, lw=1.)

    ax.axhline(0, lw=0.5, color='gray')
    ax.axvline(0, lw=0.5, color='gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{20}$", r"$0$", r"$\frac{L}{20}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$ \varepsilon(\omega) f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./SPhPPlotsSaved/wFSPhPTMPerp.png")


def createPlotTM():
    epsInf = 2.
    omega = 1. * 1e11
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    L = 0.1

    zArr = np.linspace(- L / 20., L / 20., 1000)

    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")

    checkNormalizations(allowedKs, L, omega, wLO, wTO, epsInf)
    plotWaveFunctionPara(allowedKs[:], zArr, L, omega, wLO, wTO, epsInf)
    plotWaveFunctionPerp(allowedKs[:], zArr, L, omega, wLO, wTO, epsInf)
