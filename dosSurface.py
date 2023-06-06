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

c = 3 * 1e8 * 1e6 * 1e-12

wLO = 3. * 1e12 * 1e-12
wTO = 1. * 1e12 * 1e-12

def broadDelta(x, d):
    return 1. / np.pi * (0.5 * d) / (x ** 2 + (0.5 * d) ** 2)

def epsW (w):
    return 1 - (wLO**2 - wTO**2) / (w**2 - wTO**2)

def wSPP(q):
    return np.sqrt(c**2 * q**2 + 0.5 * wLO**2 - np.sqrt(c**4 * q**4 + 0.25 * wLO**4 - c**2 * q**2 * wTO**2))

def rhoW(z, w):

    rho0 = 3.

    discretizeAngle = np.linspace(0, np.pi / 2., 10000)
    integral = (np.sin(discretizeAngle) - 0.5 * np.sin(discretizeAngle)**3) * np.sin(w/c * np.cos(discretizeAngle) * z)**2
    integalPerformed = np.mean(integral)
    return rho0 * integalPerformed


def rhoWAsOfZ(zArr, w):

    res = np.zeros(zArr.shape[0])
    for zInd, zVal in enumerate(zArr):
        res[zInd] = rhoW(zVal, w)
    return res


def rhoSPPAsOfZ(zArr, w):

    prefac1 = np.pi / 4. * np.sqrt(np.abs(epsW(w))) / np.sqrt(1 + epsW(w)**2)
    prefacNum = np.sqrt(wLO**2 - w**2) * np.sqrt(w**2 * (wLO**2 + wTO**2 - 2 * w**2) + wLO**4 - wTO**4)
    prefacDenom = np.sqrt(wLO**2 + wTO**2 - 2. * w**2)**3
    expFac = np.exp(-2. * np.sqrt(1. / np.abs(epsW(w))) * np.sqrt(wLO**2 - w**2) / (wLO**2 + wTO**2 - 2 * w**2) * w / c * zArr)

    return prefac1 * prefacNum / prefacDenom * expFac

def rhoSPPAsOfWAndZ(w, z):

    rho0 = w**2 / (np.pi**2 * c**3)
    prefac1 = np.pi / 4. * np.sqrt(np.abs(epsW(w))) / np.sqrt(1 + epsW(w)**2)
    prefacNum = np.sqrt(wLO**2 - w**2) * np.sqrt(w**2 * (wLO**2 + wTO**2 - 2 * w**2) + wLO**4 - wTO**4)
    prefacDenom = np.sqrt(wLO**2 + wTO**2 - 2. * w**2)**3
    expFac = np.exp(-2. * np.sqrt(1. / np.abs(epsW(w))) * np.sqrt(wLO**2 - w**2) / (wLO**2 + wTO**2 - 2 * w**2) * w / c * z)

    return rho0 * prefac1 * prefacNum / prefacDenom * expFac

def rhoSPPAsOfWAndZOverRho0(w, z):

    #rho0 = w**2 / (np.pi**2 * c**3)
    prefac1 = np.pi / 4. * np.sqrt(np.abs(epsW(w))) / np.sqrt(1 + epsW(w)**2)
    prefacNum = np.sqrt(wLO**2 - w**2) * np.sqrt(w**2 * (wLO**2 + wTO**2 - 2 * w**2) + wLO**4 - wTO**4)
    prefacDenom = np.sqrt(wLO**2 + wTO**2 - 2. * w**2)**3
    expFac = np.exp(-2. * np.sqrt(1. / np.abs(epsW(w))) * np.sqrt(wLO**2 - w**2) / (wLO**2 + wTO**2 - 2 * w**2) * w / c * z)

    return prefac1 * prefacNum / prefacDenom * expFac




def plotrhoAsOfz():
    wMax = 1. / np.sqrt(2) * np.sqrt(wLO**2 + wTO**2)
    print("wMax = {}".format(wMax))
    w = (wMax - 0.1) * 1e12 * 1e-12

    #zArr = np.linspace(1e-6, 1e-3, 100)
    xLow = 1e-3
    xHigh = 1e3
    zArr = np.logspace(np.log10(xLow), np.log10(xHigh), 100)
    rho = rhoSPPAsOfZ(zArr, w)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(zArr, rho, color = 'indianred', lw = 1.2)
    ax.axhline(1., color = 'gray', lw = 0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(xLow, xHigh)

    ax.set_xlabel(r"$z[\mu\rm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")


    plt.savefig("./savedPlots/fig1.png")


def plotrhoIntAsOfZ():
    wMax = 1. / np.sqrt(2) * np.sqrt(wLO**2 + wTO**2)
    print("wMax = {}".format(wMax))
    w = (wMax - 0.1) * 1e12 * 1e-12

    #zArr = np.linspace(1e-6, 1e-3, 100)
    xLow = 1e-2
    xHigh = 1e3
    zArr = np.logspace(np.log10(xLow), np.log10(xHigh), 100)
    rhoInt = np.zeros(zArr.shape[0])
    for zInd, zVal in enumerate(zArr):
        #rhoWArr = rhoSPPAsOfWAndZ(np.linspace(wTO, wLO, 10000), zVal)
        rhoInt[zInd] = quad(rhoSPPAsOfWAndZOverRho0, wTO, wMax, args=(zVal))[0]

    #rhoInt = rhoInt * 3 * 3 * np.pi**2 * c**3 / (wLO**2 - wTO**2)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(zArr, rhoInt / (wLO - wTO), color = 'indianred', lw = 1.2)
    ax.axhline(1., color = 'gray', lw = 0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(xLow, xHigh)

    ax.set_xlabel(r"$z[\mu\rm{m}]$")
    ax.set_ylabel(r"$\frac{\int_{\omega_{\rm{TO}}}^{\omega_{\rm{LO}}}\rho / \rho_0}{\omega_{\rm LO} - \omega_{\rm TO}}$")


    plt.savefig("./savedPlots/plotRhoIntFreq.png")


def plotRhoFreqs():
    wMax = 1. / np.sqrt(2) * np.sqrt(wLO**2 + wTO**2)
    print("wMax = {}".format(wMax))

    #wArr = np.linspace(wTO, wMax - 0.0001, 10)
    wArr = np.logspace(np.log10(wTO), np.log10(wMax - 1e-12), 10)

    logDistArr = np.logspace(-10, np.log10(wMax - wTO - 1e-8), 7)
    wArr = wMax - logDistArr

    xLow = 1e-3
    xHigh = 1e3
    zArr = np.logspace(np.log10(xLow), np.log10(xHigh), 100)
    rhoArr = np.zeros((wArr.shape[0], zArr.shape[0]))
    for wInd, wVal in enumerate(wArr):
        rhoArr[wInd, :] = rhoSPPAsOfZ(zArr, wVal)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapBone = cm.get_cmap('bone')
    cmapPink = cm.get_cmap('pink')

    for wInd, wVal in enumerate(wArr):
        color = cmapPink(wInd / (wArr.shape[0] + 1.))
        if(wInd == wArr.shape[0] - 1):
            ax.plot(zArr, rhoArr[wInd, :], color=color, lw=1.2,
                    label=r"$\omega_{\rm{TO}} + \varepsilon$")
            continue

        ax.plot(zArr, rhoArr[wInd, :], color = color, lw = 1.2, label = "$\omega_{\infty} - $"+"{0:.3g}".format(logDistArr[wInd]) + r"$\rm{THz}$")

    ax.axhline(1., color = 'gray', lw = 0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylim(1e-4, 1e6)

    ax.set_xlim(xLow, xHigh)

    ax.set_xlabel(r"$z[\mu\rm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=fontsize - 4, loc='upper right', bbox_to_anchor=(1.0, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    plt.savefig("./savedPlots/fig2.png")

def plotRhoZs():
    wMax = 1. / np.sqrt(2) * np.sqrt(wLO**2 + wTO**2)
    print("wMax = {}".format(wMax))

    wMax = 1. / np.sqrt(2) * np.sqrt(wLO**2 + wTO**2)

    wArr = np.linspace(wTO, wMax, 100000)
    xLow = 1e-3
    xHigh = 1e0
    zArr = np.logspace(np.log10(xLow), np.log10(xHigh), 4)
    rhoArr = np.zeros((wArr.shape[0], zArr.shape[0]))
    for zInd, zVal in enumerate(zArr):
        rhoArr[:, zInd] = rhoSPPAsOfWAndZOverRho0(wArr, zVal)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapBone = cm.get_cmap('bone')
    cmapPink = cm.get_cmap('pink')

    for zInd, zVal in enumerate(zArr):
        color = cmapPink(zInd / (zArr.shape[0] + 1.))
        ax.plot(wArr, rhoArr[:, zInd], color = color, lw = 1.2, label = "z = {}".format(int(zVal * 1000)) + r"$\rm{nm}$")

    ax.axhline(1., color = 'gray', lw = 0.5)
    ax.axvline(wTO, color = 'gray', lw = 0.5, zorder = -666)
    ax.axvline(wMax, color = 'gray', lw = 0.5, zorder = -666)

    plt.text(0.22, 0.925, r"$\omega_{\rm TO}$", fontsize=8, transform=plt.gcf().transFigure)
    plt.text(0.9, 0.925, r"$\omega_{\infty}$", fontsize=8, transform=plt.gcf().transFigure)

    #ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylim(1e-3, 1e7)
    #ax.set_xlim(xLow, xHigh)

    ax.set_xlabel(r"$\omega[\rm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=fontsize - 2, loc='upper left', bbox_to_anchor=(.1, 1.), edgecolor='black', ncol=2)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    plt.savefig("./savedPlots/rhoAsOfW.png")