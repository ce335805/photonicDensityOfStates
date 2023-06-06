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
import scipy

import dosFresnel
import findAllowedKs
import dosBox
import evanescentWavefunction

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


def plotDos(zArr, dos):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(zArr, dos, color='indianred', lw=1.)
    ax.axhline(0.333333, lw = 0.5, color = 'gray', zorder = -666)

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xlabel(r"$z[\frac{\omega}{c}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.savefig("./savedPlots/dosFresnel.png")

def plotDosCompare(zArr, dos1, dos2):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(zArr, dos1, color='peru', lw=1., label = "DOS from Box")
    ax.plot(zArr, dos2, color='teal', lw=1., label = "DOS from Fresnel")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.25, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_ylim(0., 0.69)

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    plt.savefig("./savedPlots/dosCompare.png")


def boxDosFromIntegralIntegrand(kVal, z, L, omega, eps, c):
    prefac = omega / np.pi**2 / c**2 / (omega**2 / (np.pi**2 * c**3))
    kD = np.sqrt((eps - 1) * omega ** 2 / c ** 2  + kVal ** 2.)
    num = np.sin(kVal * (L / 2. - z))**2 * np.sin(kD * L / 2.)**2
    denom = np.sin(kD * L / 2.)**2 + np.sin(kVal * L / 2.)**2
    return prefac * num / denom


def plotIntegradBox(z, L, omega, eps, c):
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    kArr = np.linspace(0., 1.1 * omega / c * L, 100000)
    integrand = boxDosFromIntegralIntegrand(kArr, z, L, omega, eps, c)

    ax.plot(kArr, integrand, color='peru', lw=1., label="DOS from Box")

    plt.savefig("./savedPlots/IntegrandBoxDos.png")

def boxDosFromIntegral(zArr, L, omega, eps, c):
    dos = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        integral = integrate.quad(boxDosFromIntegralIntegrand, 0, omega / c, args=(zVal, L, omega, eps, c))
        dos[zInd] = integral[0]
    return dos


def main():
    #computeDosFromFresnel()

    evanescentWavefunction.createPlotEva()
    exit()

    epsilon = 2.
    omega = 2 * 1e11
    c = 3 * 1e8
    L = 10.

    zArr = np.linspace(-c / omega * 20., c / omega * 40., 1000)

    dosInBoxTE = dosBox.computeDosBoxTE(zArr, L, omega, epsilon)
    dosInBoxTM = dosBox.computeDosBoxTM(zArr, L, omega, epsilon)
    print(dosInBoxTE[900])
    print(dosInBoxTM[900])
    plotDosCompare(zArr, dosInBoxTE, dosInBoxTM)

main()