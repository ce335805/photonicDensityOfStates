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

import findAllowedKs

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


def NormSqr(kDVal, L, omega, eps):
    epsNorm = 1.
    kReal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kDVal ** 2)
    denom = np.sinh(kReal * L / 2.) ** 2 * np.sin(kDVal * L / 2) ** 2
    bracket1 = omega**2 / consts.c**2 / kReal**2 * (1. + np.sinh(kReal * L) / (kReal * L) * (1. + 2. * consts.c**2 * kReal**2 / omega**2))
    bracket2 = eps * omega**2 / consts.c**2 / kDVal**2 * (1. + np.sin(kDVal * L) / (kDVal * L) * (1. - 2. * consts.c**2 * kDVal**2 / omega**2 / eps))
    bracket = epsNorm * np.sinh(kReal * L / 2)**2 * bracket2 + np.sin(kDVal * L / 2)**2 * bracket1
    return L ** 3 / 4 * bracket / denom

def waveFunctionPosParallel(zArr, kDVal, L, omega, eps):
    kReal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kDVal ** 2)
    NSqr = NormSqr(kDVal, L, omega, eps)
    func = - 1. / np.tanh(kReal * L / 2.) * np.sinh(kReal * zArr) + np.cosh(kReal * zArr)
    return 1. / np.sqrt(NSqr) * func

def waveFunctionPosPerp(zArr, kDVal, L, omega, eps):
    kReal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kDVal ** 2)
    NSqr = NormSqr(kDVal, L, omega, eps)
    func = ( - 1. / np.tanh(kReal * L / 2.) * np.cosh(kReal * zArr) + np.sinh(kReal * zArr))
    return 1. / np.sqrt(NSqr) * np.sqrt(omega**2 / consts.c**2 / kReal**2 + 1) * func

def waveFunctionNegParallel(zArr, kDVal, L, omega, eps):
    NSqr = NormSqr(kDVal, L, omega, eps)
    func = 1. / np.tan(kDVal * L / 2.) * np.sin(kDVal * zArr) + np.cos(kDVal * zArr)
    return 1. / np.sqrt(NSqr) * func

def waveFunctionNegPerp(zArr, kDVal, L, omega, eps):
    NSqr = NormSqr(kDVal, L, omega, eps)
    func = ( - 1. / np.tan(kDVal * L / 2.) * np.cos(kDVal * zArr) + np.sin(kDVal * zArr))
    return 1. / np.sqrt(NSqr) * np.sqrt(eps * omega**2 / kDVal**2 / consts.c**2 - 1) * func


def waveFunctionTMEvaParallel(zArr, kArr, L, omega, eps):

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPosParallel(zPosArr, kArr, L, omega, eps)
    wFNeg = waveFunctionNegParallel(zNegArr, kArr, L, omega, eps)

    wF = np.append(wFNeg, wFPos)

    return wF

def waveFunctionTMEvaPerp(zArr, kArr, L, omega, eps):

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPosPerp(zPosArr, kArr, L, omega, eps)
    wFNeg = waveFunctionNegPerp(zNegArr, kArr, L, omega, eps)

    wF = np.append(wFNeg, wFPos)

    return wF

def waveFunctionForInt(z, k, L, omega, eps):

    if(z >= 0):
        return (waveFunctionPosParallel(z, k, L, omega, eps))**2 + (waveFunctionPosPerp(z, k, L, omega, eps))**2
    else:
        return (waveFunctionNegParallel(z, k, L, omega, eps))**2 + (waveFunctionNegPerp(z, k, L, omega, eps))**2


def findKsTMEva(L, omega, eps):
    #Factor of 10 for more points and a +17 to avoid special numbers
    NDiscrete = 10 * int(omega / consts.c * L / (4. * np.pi) + 17)
    print("NDiscrete = {}".format(NDiscrete))

    extremaTMEva = findAllowedKs.extremalPoints(L, omega, eps, NDiscrete, "TMEva")
    findAllowedKs.plotRootFuncWithExtrema(L, omega, eps, extremaTMEva, "TMEva")
    rootsTMEva = findAllowedKs.computeRootsGivenExtrema(L, omega, eps, extremaTMEva, "TMEva")
    print("Number of roots for TM found = {}".format(rootsTMEva.shape))
    findAllowedKs.plotRootFuncWithRoots(L, omega, eps, rootsTMEva, "TMEva")
    return rootsTMEva

def checkNormalizationK(k, L, omega, eps):

    checkInt = integrate.quad(waveFunctionForInt, -L / 2., L / 2., args=(k, L, omega, eps))
    print("Norm = {}, with Int accuracy = {}".format(checkInt[0] * L**2, checkInt[1]))

def checkNormalizations(L, omega, eps):
    allowedKs = findKsTMEva(L, omega, eps)
    if(allowedKs[0] == 0):
        allowedKs = allowedKs[1:]
    for kVal in allowedKs[:10]:
        checkNormalizationK(kVal, L, omega, eps)


def plotWaveFunction(kDArr, zArr, L, omega, eps):

    wFParallel = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)
    wFPerp = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)

    for kDInd, kDVal in enumerate(kDArr):
        wFParallel[kDInd, :] = waveFunctionTMEvaParallel(zArr, kDVal, L, omega, eps)
        wFPerp[kDInd, :] = waveFunctionTMEvaPerp(zArr, kDVal, L, omega, eps)

    zeroInd = np.where(zArr < 0)
    wFPerp[:, zeroInd] = eps * wFPerp[:, zeroInd]

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    for kDInd, kDVal in enumerate(kDArr):
        color1 = cmapPink(kDInd / (wFParallel.shape[0] + 0.5))
        color2 = cmapBone(kDInd / (wFParallel.shape[0] + 0.5))
        ax.plot(zArr, wFParallel[kDInd, :], color=color1, lw=1.)
        #ax.plot(zArr, wFPerp[kDInd, :], color=color2, lw=1.)

    ax.axhline(0, lw = 0.5, color = 'gray')
    ax.axvline(0, lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./savedPlots/wFTMEvaPara.png")

def createPlotTM():

    epsilon = 2.
    omega = 2 * 1e11
    c = 3 * 1e8
    L = 0.05

    checkNormalizations(L, omega, epsilon)

    zArr = np.linspace(-c / omega * 10., c / omega * 20., 1000)
    zArr = np.linspace(- L / 2., L / 2., 1000)

    allowedKs = findKsTMEva(L, omega, epsilon)

    print(allowedKs[0 : 10])
    plotWaveFunction(allowedKs[0:4], zArr, L, omega, epsilon)

