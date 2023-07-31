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

def plotDosAsOfZ(zArr, dosFP, omega, L):

    rho0 = omega**2 / np.pi**2 / 3. / consts.c**3

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(zArr, dosFP / rho0, color='indianred', lw=1.)
    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./FPStuff/FPPlotsSaved/dosFPAsOfZ.png")

def plotDosAsOfZSeveralOm(dosFP, zArr, omArr, L):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.8, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    for omInd, omVal in enumerate(omArr):
        color = cmapPink(omInd / (omArr.shape[0] + 0.5))
        rho0 = omVal ** 2 / np.pi ** 2 / 3. / consts.c ** 3
        ax.plot(zArr * 1e6, dosFP[:, omInd] / rho0, color=color, lw=1., label = r"{}".format(int(omVal * 1e-12)) + r"$\rm{THz}$")

    ax.set_xlim(np.amin(zArr) * 1e6, np.amax(zArr) * 1e6)
    ax.axhline(1., color = 'black', lw = 0.5)

    ax.set_xlabel(r"$z \, [\mu \rm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    legend = ax.legend(fontsize=fontsize, loc='lower center', bbox_to_anchor=(0.5, 0.95), edgecolor='black', ncol=2)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig("./FPStuff/FPPlotsSaved/dosFPAsOfZOmegas.png")


def plotDosAsOfOm(omArr, dosFP, L):

    rho0 = omArr**2 / np.pi**2 / 3. / consts.c**3

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(omArr, dosFP / rho0, color='indianred', lw=1.)
    ax.set_xlim(np.amin(omArr), np.amax(omArr))
    ax.axhline(1., color = 'black', lw = .5, zorder = -666)
    ax.set_xscale("log")

    ax.set_xlabel(r"$\omega \, \left[\frac{1}{s}\right]$")
    ax.set_ylabel(r"$\rho / \rho_0$")


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./FPStuff/FPPlotsSaved/dosFPAsOfOm.png")

