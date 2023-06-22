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


def XBracket(kDArr, L, omega, eps, epsNorm):
    kArr = np.sqrt((eps - 1) * omega ** 2 / consts.c ** 2 - kDArr[None, :] ** 2)
    term1 = np.sin(kDArr * L / 2)**2 * omega**2 / consts.c**2 / kArr**2 * (1 + (1 + 2 * consts.c**2 * kArr **2 / omega**2) * np.sinh(kArr * L) / kArr / L)
    #term1 = np.sin(kDArr * L / 2)**2 * omega**2 / consts.c**2 / kArr**2
    term2 = epsNorm * np.sinh(kArr * L / 2)**2 * eps * omega ** 2 / consts.c**2 / kDArr**2 * (1 + (1 - 2 * consts.c**2 * kDArr**2 / eps / omega**2) * np.sin(kDArr**2 * L) / kDArr / L)
    #term2 = epsNorm * np.sinh(kArr * L / 2)**2 * eps * omega ** 2 / consts.c**2 / kDArr**2
    return term1 + term2

def dosEvaPosTM(zArr, kDArr, L, omega, eps):
    c = consts.c
    epsNorm = 1.
    kArr = np.sqrt((eps - 1) * omega ** 2 / c ** 2 - kDArr[None, :] ** 2)
    prefac = 2. * np.pi * c / omega / L

    num1 = np.sinh(kArr * (L / 2 - zArr[:, None]))**2
    num2 = (omega**2 / consts.c**2 / kArr**2 - 1.) * np.cosh(kArr * (L / 2 - zArr[:, None]))**2
    num = np.sin(kDArr * L / 2.)**2 * (num1 + num2)
    denom = XBracket(kDArr, L, omega, eps, epsNorm)

    dos = prefac * num / denom
    dosSummed = np.sum(dos, axis=1)
    return dosSummed

def dosEvaNegTM(zArr, kDArr, L, omega, eps):
    c = consts.c
    epsNorm = 1.
    kArr = np.sqrt((eps - 1) * omega ** 2 / c ** 2 - kDArr[None, :] ** 2)
    prefac = 2. * eps * np.pi * c / omega / L

    num1 = np.sin(kDArr * (L / 2 + zArr[:, None]))**2
    num2 = (eps * omega**2 / consts.c**2 / kDArr[None, :]**2 - 1.) * np.cos(kDArr * (L / 2 + zArr[:, None]))**2
    num = np.sinh(kArr * L / 2.)**2 * (num1 + num2)
    denom = XBracket(kDArr, L, omega, eps, epsNorm)

    dos = prefac * num / denom
    dosSummed = np.sum(dos, axis=1)
    return dosSummed



def computeDosTMEva(z, L, omega, eps):

    #Factor of 10 for more points and a +17 to avoid special numbers
    NDiscrete = 10 * int(omega / consts.c * L / (4. * np.pi) + 17)
    print("NDiscrete = {}".format(NDiscrete))

    extremaTMEva = findAllowedKs.extremalPoints(L, omega, eps, NDiscrete, "TMEva")
    findAllowedKs.plotRootFuncWithExtrema(L, omega, eps, extremaTMEva, "TMEva")
    rootsTMEva = findAllowedKs.computeRootsGivenExtrema(L, omega, eps, extremaTMEva, "TMEva")
    print("Number of roots for TMEva found = {}".format(rootsTMEva.shape))
    findAllowedKs.plotRootFuncWithRoots(L, omega, eps, rootsTMEva, "TMEva")

    #exit()

    indNeg = np.where(z < 0)
    indPos = np.where(z >= 0)
    zPosArr = z[indPos]
    zNegArr = z[indNeg]

    dosTMEvaPos = dosEvaPosTM(zPosArr, rootsTMEva, L, omega, eps)
    dosTMEvaNeg = dosEvaNegTM(zNegArr, rootsTMEva, L, omega, eps)
    dosTMEva = np.append(dosTMEvaNeg, dosTMEvaPos)
    return dosTMEva

