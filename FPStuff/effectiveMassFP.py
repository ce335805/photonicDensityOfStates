import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad
import math

from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import h5py
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch

import TMEvaWaveFunction
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.constants as consts

import dosFresnel
import findAllowedKs
import dosBox
import TEEvaWaveFunction
import dosEvanescentTE
import dosEvanescentTM
import TEWaveFunction
import TMWaveFunctions

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

import computeDosFP

def plotEffectiveMassIntegrad(L, omArr):

    integrand = diffIntegrand(L, omArr)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(omArr, integrand, color='indianred', lw=1.)
    ax.set_xlim(np.amin(omArr), np.amax(omArr))
    ax.set_xscale("log")

    ax.set_xlabel(r"$\omega \, [\frac{1}{s}]$")
    ax.set_ylabel(r"$ \frac{d}{d \Lambda} (\Delta t / t_0)$")

    plt.savefig("./FPStuff/FPPlotsSaved/integrandEffMass.png")

def smoothenFunction(data):
    smoothDegree = math.floor(len(data) * 0.0002)
    smoothDegree = 0
    smoothData = data
    for smoothD in np.arange(smoothDegree):
        smoothData += np.roll(data, -smoothD + 1) + np.roll(data, smoothD + 1)
    return smoothData / (2 * smoothDegree + 1)


def diffIntegrand(L, omArr):
    alat = 1e-10

    dosFree = omArr**2 / np.pi**2 / consts.c**3 / 3.
    print("computed DOS Free")
    dosFPAsOfOm = computeDosFP.computeDosFPAsOfFeq(L / 2., omArr, L)
    print("computed Dos in FP")
    prefac = 2. * consts.fine_structure * alat**2 * omArr / 3. / np.pi / consts.c**2
    toInt = prefac * (dosFPAsOfOm / dosFree - 1)
    #toInt = np.sqrt(omArr)**2 * (dosFPAsOfOm / dosFree - 1)
    toInt = smoothenFunction(toInt)

    return toInt

def integrateDosDifference(L, omArr):
    toInt = diffIntegrand(L, omArr)
    return np.trapz(toInt, omArr)

