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
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.constants as consts

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

import dosFuncs.dosTEModes as dosTE
import dosFuncs.dosTEEvaModes as dosTEEva
import dosFuncs.dosTMModes as dosTM
import dosFuncs.dosTMEvaModes as dosTMEva
import epsilonFunctions as epsFunc

def plotTEWhole():
    omega = 6 * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = .1

    zArr = np.linspace(-L / 2., L / 2., 500)

    dosTEPart = dosTE.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    dosTEEvaPart = dosTEEva.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)

    createPlotDos2Combined(zArr, dosTEPart, dosTEEvaPart, L, omega, wLO, wTO, epsInf)

def plotTMWhole():
    omega = 1 * 1e11
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = .2

    zArr = np.linspace(-L / 2., L / 2., 500)

    dosTMPart = dosTM.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    dosTMEvaPart = dosTMEva.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)

    createPlotDos2Combined(zArr, dosTMPart, dosTMEvaPart, L, omega, wLO, wTO, epsInf)


def createPlotDos2Combined(zArr, dos1, dos2, L, omega, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(zArr, dos1, color=cmapPink(0.3), lw=1.)
    ax.plot(zArr, dos2, color=cmapPink(0.6), lw=1.)
    ax.plot(zArr, dos1 + dos2, color=cmapBone(0.45), lw=1.)
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([- L / 2., 0., L / 2.])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosTMWhole.png")

