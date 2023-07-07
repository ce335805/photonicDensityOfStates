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
    kDVal = epsFunc.kDFromKEva(kVal, omega, wLO, wTO, epsInf)
    brack1 = (1 - np.exp(- 2. * kVal * L)) / (2. * kVal * L) - np.exp(- kVal * L)
    term1 = brack1 * np.sin(kDVal * L / 2) ** 2
    brack2 = 1 - np.sin(kDVal * L) / (kDVal * L)
    term2 = brack2 * 0.25 * (1 + np.exp(-2 * kVal * L) - 2. * np.exp(- kVal * L))
    return L / 4 * (term1 + term2)

def waveFunctionPos(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKEva(kVal, omega, wLO, wTO, epsInf)
    func = 0.5 * (np.exp(- kVal * zArr) - np.exp(kVal * zArr - kVal * L)) * np.sin(kDVal * L / 2.)
    return 1. / np.sqrt(NSqr) * func

def waveFunctionNeg(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKEva(kVal, omega, wLO, wTO, epsInf)
    func = np.sin(kDVal * (L / 2. + zArr)) * 0.5 * (1 - np.exp(- kVal * L))
    return 1. / np.sqrt(NSqr) * func

def waveFunctionForInt(z, k, L, omega, wLO, wTO, epsInf):
    if (z >= 0):
        return (waveFunctionPos(z, k, L, omega, wLO, wTO, epsInf)) ** 2
    else:
        return (waveFunctionNeg(z, k, L, omega, wLO, wTO, epsInf)) ** 2

def checkNormalizationK(k, L, omega, wLO, wTO, epsInf):
    checkInt = integrate.quad(waveFunctionForInt, -L / 2., L / 2., args=(k, L, omega, wLO, wTO, epsInf))
    print("Norm = {}, with Int accuracy = {}".format(checkInt[0], checkInt[1]))

def checkNormalizations(allowedKs, L, omega, wLO, wTO, epsInf):
    for kVal in allowedKs[:10]:
        checkNormalizationK(kVal, L, omega, wLO, wTO, epsInf)


def waveFunctionTEEva(zArr, kArr, L, omega, wLO, wTO, epsInf):
    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPos(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
    wFNeg = waveFunctionNeg(zNegArr, kArr, L, omega, wLO, wTO, epsInf)

    wF = np.append(wFNeg, wFPos)

    return wF

def plotWaveFunction(kArr, zArr, L, omega, wLO, wTO, epsInf):
    wF = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)

    for kInd, kVal in enumerate(kArr):
        wF[kInd, :] = waveFunctionTEEva(zArr, kVal, L, omega, wLO, wTO, epsInf)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    for kInd, kVal in enumerate(kArr):
        color = cmapPink(kInd / (wF.shape[0] + 0.5))
        ax.plot(zArr, wF[kInd, :], color=color, lw=1.)

    ax.axhline(0, lw=0.5, color='gray')
    ax.axvline(0, lw=0.5, color='gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    #ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    #ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./SPhPPlotsSaved/wFSPhPTEEva.png")


def createPlotTEEva():
    epsInf = 2.
    omega = 5. * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    L = 0.05

    zArr = np.linspace(-consts.c / omega * 10., consts.c / omega * 10., 1000)
    zArr = np.linspace(- L / 2., L / 2., 1000)


    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TEEva")
    checkNormalizations(allowedKs, L, omega, wLO, wTO, epsInf)
    plotWaveFunction(allowedKs[-3:], zArr, L, omega, wLO, wTO, epsInf)
