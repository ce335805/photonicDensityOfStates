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


def NormSqr(kVal, L, omega, eps):
    epsNorm = 1.
    kDVal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)
    denom = np.sin(kVal * L / 2.) ** 2 * np.sin(kDVal * L / 2) ** 2
    bracket = epsNorm * (1 - np.sin(kDVal * L) / (kDVal * L)) * np.sin(kVal * L / 2.) ** 2 + (
                1 - np.sin(kVal * L) / (kVal * L)) * np.sin(kDVal * L / 2.) ** 2
    return L ** 3 / 4 * bracket / denom


def waveFunctionPos(zArr, kVal, L, omega, eps):
    NSqr = NormSqr(kVal, L, omega, eps)
    func = - 1. / np.tan(kVal * L / 2.) * np.sin(kVal * zArr) + np.cos(kVal * zArr)
    return 1. / np.sqrt(NSqr) * func


def waveFunctionNeg(zArr, kVal, L, omega, eps):
    NSqr = NormSqr(kVal, L, omega, eps)
    kDVal = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)
    func = 1. / np.tan(kDVal * L / 2.) * np.sin(kDVal * zArr) + np.cos(kDVal * zArr)
    return 1. / np.sqrt(NSqr) * func


def waveFunctionTE(zArr, kArr, L, omega, eps):
    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    wFPos = waveFunctionPos(zPosArr, kArr, L, omega, eps)
    wFNeg = waveFunctionNeg(zNegArr, kArr, L, omega, eps)

    wF = np.append(wFNeg, wFPos)

    return wF


def waveFunctionForInt(z, k, L, omega, eps):
    if (z >= 0):
        return (waveFunctionPos(z, k, L, omega, eps)) ** 2
    else:
        return eps * (waveFunctionNeg(z, k, L, omega, eps)) ** 2


def waveFunctionScalarProduct(z, k1, k2, L, omega, eps):
    if (z >= 0):
        return waveFunctionPos(z, k1, L, omega, eps) * waveFunctionPos(z, k2, L, omega, eps)
    else:
        return waveFunctionNeg(z, k1, L, omega, eps) * waveFunctionNeg(z, k2, L, omega, eps)


def findKsTE(L, omega, eps):
    # Factor of 10 for more points and a +17 to avoid special numbers
    NDiscrete = 10 * int(omega / consts.c * L / (4. * np.pi) + 17)
    print("NDiscrete = {}".format(NDiscrete))

    extremaTE = findAllowedKs.extremalPoints(L, omega, eps, NDiscrete, "TE")
    # findAllowedKs.plotRootFuncWithExtrema(L, omega, eps, extremaTEEva, "TEEva")
    rootsTE = findAllowedKs.computeRootsGivenExtrema(L, omega, eps, extremaTE, "TE")
    print("Number of roots for TEE found = {}".format(rootsTE.shape))
    # findAllowedKs.plotRootFuncWithRoots(L, omega, eps, rootsTEEva, "TEEva")
    return rootsTE


def checkNormalizationK(k, L, omega, eps):
    checkInt = integrate.quad(waveFunctionForInt, -L / 2., L / 2., args=(k, L, omega, eps))
    print("Norm = {}, with Int accuracy = {}".format(checkInt[0] * L ** 2, checkInt[1]))


def checkNormalizations(L, omega, eps):
    allowedKs = findKsTE(L, omega, eps)
    for kVal in allowedKs[:10]:
        checkNormalizationK(kVal, L, omega, eps)


def orthonormalityIntegral(k1, k2, L, omega, eps):
    checkInt = integrate.quad(waveFunctionScalarProduct, -L / 2., L / 2., args=(k1, k2, L, omega, eps))
    return checkInt[0] * L**2

def checkOrthonormality(L, omega, eps):
    allowedKs = findKsTE(L, omega, eps)
    scalarProducts = np.zeros((len(allowedKs), len(allowedKs)))
    for k1Ind, k1Val in enumerate(allowedKs):
        for k2Ind, k2Val in enumerate(allowedKs):
            scalarProducts[k1Ind, k2Ind] = orthonormalityIntegral(k1Val, k2Val, L, omega, eps)

    return scalarProducts


def plotWaveFunction(kDArr, zArr, L, omega, eps):
    wF = np.zeros((kDArr.shape[0], zArr.shape[0]), dtype=float)

    for kDInd, kDVal in enumerate(kDArr):
        wF[kDInd, :] = waveFunctionTE(zArr, kDVal, L, omega, eps)

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
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$f(z) \; [\mathrm{arb. \, units}]$")

    plt.savefig("./savedPlots/wFTE.png")


def createPlotTE():
    epsilon = 2.
    omega = 1 * 1e11
    c = 3 * 1e8
    L = 0.1

    checkNormalizations(L, omega, epsilon)
    orthomat = checkOrthonormality(L, omega, epsilon)

    print(orthomat)

    zArr = np.linspace(-c / omega * 40., c / omega * 20., 1000)
    zArr = np.linspace(- L / 2., L / 2., 1000)

    allowedKs = findKsTE(L, omega, epsilon)

    print(allowedKs)
    plotWaveFunction(allowedKs[:], zArr, L, omega, epsilon)
