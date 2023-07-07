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


def NormSqr(kArr, L, omega, wLO, wTO, epsInf):
    kDArr = epsFunc.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    brack11 = np.exp(- kArr * L) * omega**2 / (consts.c**2 * kArr**2)
    brack12 = (omega**2 / (consts.c**2 * kArr**2) + 2) * (1 - np.exp(- 2 * kArr * L)) / (2 * kArr * L)
    term1 = (brack11 + brack12) * 0.25 * (1 + np.exp(-2. * kDArr * L) - 2. * np.exp(-kDArr * L))

    brack21 = np.exp(- kDArr * L) * eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2)
    brack22 = (eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2) + 2) * (1 - np.exp(- 2 * kDArr * L)) / (2 * kDArr * L)
    term2 = (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kArr * L) - 2. * np.exp(-kArr * L))

    return L / 4. * (term1 + term2)

def waveFunctionPosPara(zArr, kArr, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
    func = 0.25 * (np.exp(- kArr[None, :] * zArr[:, None]) - np.exp(kArr[None, :] * (zArr[:, None] - L))) * (1 - np.exp(-kDArr[None, :] * L))
    return np.sum(1. / NSqr[None, :] * func ** 2, axis=1)


def waveFunctionNegPara(zArr, kArr, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
    func = 0.25 * (np.exp(kDArr[None, :] * zArr[:, None]) - np.exp(- kDArr[None, :] * (L + zArr[:, None]))) * (1 - np.exp(-kArr[None, :] * L))
    return np.sum(1. / NSqr[None, :] * func ** 2, axis=1)


def waveFunctionPosPerp(zArr, kArr, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
    func = 0.25 * np.sqrt(omega**2 / (consts.c**2 * kArr[None, :]**2) + 1) * ( np.exp(- kArr[None, :] * zArr[:, None]) +  np.exp(kArr[None, :] * (zArr[:, None] - L))) * (1 - np.exp(-kDArr[None, :] * L))
    return np.sum(1. / NSqr[None, :] * func ** 2, axis=1)


def waveFunctionNegPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    func = -0.25 * np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) + 1) * ( np.exp(kDArr[None, :] * zArr[:, None]) +  np.exp(-kDArr[None, :] * (L + zArr[:, None]))) * (1 - np.exp(-kVal * L))
    return np.sum(1. / NSqr[None, :] * func ** 2, axis=1)

def calcDosTM(zArr, L, omega, wLO, wTO, epsInf):

    kArr = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "Surf")

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = waveFunctionPosPara(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
    dosNeg = waveFunctionNegPara(zNegArr, kArr, L, omega, wLO, wTO, epsInf)
    dosPos += waveFunctionPosPerp(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
    dosNeg += waveFunctionNegPerp(zNegArr, kArr, L, omega, wLO, wTO, epsInf)

    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos

def plotDosTMSPhP(zArr, dos, L, omega, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosSurf.png")

def createPlotDosTMSurf():

    omega = 2. * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = 1.

    zArr = np.linspace(-L / 500., L / 500., 1000)

    dos = calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    plotDosTMSPhP(zArr, dos, L, omega, wLO, wTO, epsInf)



