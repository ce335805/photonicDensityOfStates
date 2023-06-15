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


def dosFuncBoxOverFreeTE(z, kArr, L, omega, eps):
    c = consts.c
    epsNorm = 1.
    kD = np.sqrt((eps - 1) * omega ** 2 / c ** 2 + kArr[None, :] ** 2)
    prefac = 2. * np.pi * c / omega / L

    num = np.sin(kArr[None, :] * (L / 2. - z[:, None])) ** 2 * np.sin(kD * L / 2.) ** 2
    denom = np.sin(kD * L / 2.) ** 2 + epsNorm * np.sin(kArr[None, :] * L / 2.) ** 2

    dos = prefac * num / denom
    dosSummed = np.sum(dos, axis=1)
    return dosSummed

def dosFuncBoxNegOverFreeTE(z, kArr, L, omega, eps):
    c = consts.c
    epsNorm = 1.
    kD = np.sqrt((eps - 1) * omega ** 2 / c ** 2 + kArr[None, :] ** 2)
    prefac = 2. * np.pi * c / omega / L

    num = np.sin(kD * (L / 2. + z[:, None])) ** 2 * np.sin( kArr[None, :] * L / 2.) ** 2
    denom = np.sin(kD * L / 2.) ** 2 + epsNorm * np.sin(kArr[None, :] * L / 2.) ** 2

    dos = prefac * num / denom
    dosSummed = np.sum(dos, axis=1)
    return dosSummed

def dosFuncBoxOverFreeTM(z, kArr, L, omega, eps):
    c = consts.c
    epsNorm = 1.
    kD = np.sqrt((eps - 1) * omega ** 2 / c ** 2 + kArr[None, :] ** 2)
    prefac = 2. * np.pi * c / omega / L


    num1 = np.sin(kArr[None, :] * (L / 2. - z[:, None])) ** 2
    num2 = (omega**2 / consts.c**2 / kArr[None, :]**2 - 1) * np.cos(kArr[None, :] * (L / 2. - z[:, None])) ** 2
    num = (num1 + num2) * np.sin( kD * L / 2.) ** 2
    normDenom1 = omega**2 / consts.c**2 / kArr[None, :]**2 *  np.sin(kD * L / 2.) ** 2
    normDenom2 = eps * omega**2 / consts.c**2 / kD**2 * epsNorm * np.sin(kArr[None, :] * L / 2.) ** 2
    denom = normDenom1 + normDenom2

    dos = prefac * num / denom
    dosSummed = np.sum(dos, axis=1)
    return dosSummed

def dosFuncBoxNegOverFreeTM(z, kArr, L, omega, eps):
    c = consts.c
    epsNorm = 1.
    kD = np.sqrt((eps - 1) * omega ** 2 / c ** 2 + kArr[None, :] ** 2)
    prefac = 2. * np.pi * c / omega / L


    num1 = np.sin(kD * (L / 2. + z[:, None])) ** 2
    num2 = (eps * omega**2 / consts.c**2 / kD**2 - 1) * np.cos(kD * (L / 2. + z[:, None])) ** 2
    num = (num1 + num2) * np.sin( kArr[None, :] * L / 2.) ** 2
    normDenom1 = omega**2 / consts.c**2 / kArr[None, :]**2 *  np.sin(kD * L / 2.) ** 2
    normDenom2 = eps * omega**2 / consts.c**2 / kD**2 * epsNorm * np.sin(kArr[None, :] * L / 2.) ** 2
    denom = normDenom1 + normDenom2

    dos = prefac * num / denom
    dosSummed = np.sum(dos, axis=1)
    return dosSummed


def computeDosBoxTE(z, L, omega, eps):

    #Factor of 10 for more points and a +17 to avoid special numbers
    NDiscrete = 10 * int(omega / consts.c * L / (4. * np.pi) + 17)
    print("NDiscrete = {}".format(NDiscrete))

    extremaTE = findAllowedKs.extremalPoints(L, omega, eps, NDiscrete, "TE")
    #findAllowedKs.plotRootFuncWithExtrema(L, omega, eps, extrema)
    rootsTE = findAllowedKs.computeRootsGivenExtrema(L, omega, eps, extremaTE, "TE")
    print("Number of roots for TE found = {}".format(rootsTE.shape))
    findAllowedKs.plotRootFuncWithRoots(L, omega, eps, rootsTE, "TE")

    #exit()

    indNeg = np.where(z < 0)
    indPos = np.where(z >= 0)
    zPosArr = z[indPos]
    zNegArr = z[indNeg]

    dosBoxPos = dosFuncBoxOverFreeTE(zPosArr, rootsTE, L, omega, eps)
    dosBoxNeg = dosFuncBoxNegOverFreeTE(zNegArr, rootsTE, L, omega, eps)
    dosBoxTE = np.append(dosBoxNeg, dosBoxPos)
    return dosBoxTE

def computeDosBoxTM(z, L, omega, eps):

    #Factor of 10 for more points and a +17 to avoid special numbers
    NDiscrete = 10 * int(omega / consts.c * L / (4. * np.pi) + 17)
    print("NDiscrete = {}".format(NDiscrete))

    extremaTM = findAllowedKs.extremalPoints(L, omega, eps, NDiscrete, "TM")
    findAllowedKs.plotRootFuncWithExtrema(L, omega, eps, extremaTM, "TM")
    rootsTM = findAllowedKs.computeRootsGivenExtrema(L, omega, eps, extremaTM, "TM")
    print("Number of roots for TM found = {}".format(rootsTM.shape))
    findAllowedKs.plotRootFuncWithRoots(L, omega, eps, rootsTM, "TM")

    indNeg = np.where(z < 0)
    indPos = np.where(z >= 0)
    zPosArr = z[indPos]
    zNegArr = z[indNeg]

    dosBoxPos = dosFuncBoxOverFreeTM(zPosArr, rootsTM, L, omega, eps)
    dosBoxNeg = dosFuncBoxNegOverFreeTM(zNegArr, rootsTM, L, omega, eps)
    dosBoxTM = np.append(dosBoxNeg, dosBoxPos)
    return dosBoxTM

