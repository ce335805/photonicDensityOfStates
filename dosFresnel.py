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


import scipy.integrate as integrate
import scipy.optimize as opt
import scipy

def integrandOsc(x, zeta, omega, eps, c):
    return 0.5 * (1. - eps) * np.cos(2. * x * zeta * omega / c) / (4. * x**2 + eps - 1.)

def intFuc1(x, eps):
    return 0.5 * (2. * x**2. + eps - 1) / (4. * x**2 + eps - 1)

def intFuc2(x, eps):
    return eps * np.sqrt(eps) * 0.5 * (2. * x**2) / (4. * x**2 + 1. / eps - 1)


def oscillatingTerm(zeta, omega, eps, c):
    oscTerm = integrate.quad(integrandOsc, 0, 1., args=(zeta, omega, eps, c))
    term1 = integrate.quad(intFuc1, 0, 1., args=(eps))
    term2 = integrate.quad(intFuc2, np.sqrt(1. - 1. / eps), 1., args=(eps))
    return term1[0] + term2[0] + oscTerm[0]
    #return 1. + oscTerm[0]


def computeDosFromFresnel(zArr, omega, eps, c):

    oscArr = np.zeros(zArr.shape[0])

    for zInd, zVal in enumerate(zArr):
        oscArr[zInd] = oscillatingTerm(zVal, omega, eps, c)

    return oscArr
