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


def plotEpsilonOmega():

    omegaArr = np.linspace(0, 5., 1000)
    wTO = 1.1
    wLO = 2.5

    eps = (wLO**2 - omegaArr**2) / (wTO**2 - omegaArr**2)
    fig = plt.figure(figsize=(2., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.92 , bottom=0.22, left=0.2, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(omegaArr, eps, lw = .7, color = cmapBone(.4))
    ax.axhline(0., color = 'black', lw = .3, linestyle = "-", zorder = -666)
    ax.axhline(1., color = 'black', lw = .3, linestyle = "-", zorder = -666)

    ymin = -3
    ymax = 7

    ax.fill_between(omegaArr, y1 = ymin, y2 = ymax, where=(omegaArr >= wTO) & (omegaArr <= wLO), color='oldlace', alpha=1., zorder = -667)


    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0., 4.)

    ax.set_ylabel(r"$\varepsilon(\omega)$", fontsize = 8)
    ax.set_xlabel(r"$\omega \, [\mathrm{arb.\,units}]$", fontsize = 8)

    ax.set_xticks([0., wTO, wLO])
    ax.set_xticklabels(["$0$", r"$\omega_{\rm TO}$", r"$\omega_{\rm LO}$"], fontsize = 8)
    ax.set_yticks([0., 1.])
    ax.set_yticklabels(["$0$", "$1$"], fontsize = 8)

    #ax.text(wTO + 0.1, 3., r"$\varepsilon(\omega) < 0$", fontsize = 8)
    ax.text(wLO + .1, 1.3, r"$\varepsilon(\omega) \to 1$", fontsize = 8)
    ax.text(wTO - 0.05, 7.2, r"$\mathrm{Restrahlenband}$", fontsize = 8)

    ax.text(-0.2, 1., r"$(\mathrm{b})$", transform = ax.transAxes, fontsize = 8)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("ThesisPlots/thesisEps.png")

def dispersionAsOfQ(wLO, wTO, Q):
    wLOSq = wLO ** 2
    wTOSq = wTO ** 2
    cqSq = consts.c**2 * Q**2

    sqrtTerm = wLOSq ** 2 + 2 * wLOSq * cqSq - 4 * wTOSq * cqSq + cqSq**2

    sol1 = .5 * (-np.sqrt(sqrtTerm) + wLOSq + cqSq)
    sol2 = .5 * (+np.sqrt(sqrtTerm) + wLOSq + cqSq)

    return (sol1, sol2)

def dispersionSurface(wLO, wTO, Q):
    wLOSq = wLO ** 2
    wTOSq = wTO ** 2
    cqSq = consts.c**2 * Q**2

    return .5 * (-np.sqrt(wLOSq**2 - 4. * wTOSq * cqSq + 4. * cqSq**2) + wLOSq + 2. * cqSq)

def plotDispersion():

    qArr = np.linspace(0., 2. * 1e4, 1000)
    disVac = consts.c * qArr
    wLO = 3 * 1e12
    wTO = 1 * 1e12
    disp1, disp2 = dispersionAsOfQ(wLO, wTO, qArr)

    wTOInd = np.where(consts.c * qArr > wTO)
    qArrSurf = qArr[wTOInd]
    dispSurf = dispersionSurface(wLO, wTO, qArrSurf)


    fig = plt.figure(figsize=(2., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.92 , bottom=0.22, left=0.25, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(consts.c * qArr, disVac, lw = .7, color = "gray", linestyle = "--")
    ax.plot(consts.c * qArr, np.sqrt(disp1), lw = .7, color = cmapBone(.4))
    ax.plot(consts.c * qArr, np.sqrt(disp2), lw = .7, color = cmapBone(.4))

    ax.plot(consts.c * qArrSurf, np.sqrt(dispSurf), lw=.7, color=cmapPink(.3))

    ax.axhline(np.sqrt((wLO**2 + wTO**2) / 2), color = "black", lw = .4)

    ax.fill_between(consts.c * qArr, y1 = wTO, y2 = wLO, color='oldlace', alpha=1.)

    ax.set_ylabel(r"$\omega(|k|)\,[\mathrm{arb.\,units}]$", fontsize = 8)
    ax.set_xlabel(r"$|k|\,[\mathrm{arb.\,units}]$", fontsize = 8)

    #ax.set_ylim(ymin, ymax)
    ax.set_xlim(consts.c * np.amin(qArr), consts.c * np.amax(qArr))
    ax.set_ylim(consts.c * np.amin(qArr), consts.c * np.amax(qArr))

    ax.set_yticks([0., wTO, np.sqrt((wLO**2 + wTO**2) / 2), wLO])
    ax.set_yticklabels(["$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)
    ax.set_xticks([0, wTO, wLO])
    ax.set_xticklabels(["$0$", r"$\frac{\omega_{\rm TO}}{c}$", r"$\frac{\omega_{\rm LO}}{c}$"], fontsize = 8)

    ax.text(.1 * wTO, 1.5 * wLO, r"$\varepsilon(\omega) > 0$", fontsize = 8)
    ax.text(1.3 * wLO, 0.3 * wTO, r"$\varepsilon(\omega) > 0$", fontsize = 8)
    ax.text(1.3 * wLO, 1.2 * wTO, r"$\varepsilon(\omega) < 0$", fontsize = 8)
    ax.text(1.7 * wLO, 1.5 * wLO, r"$c|k|$", fontsize = 8)

    ax.text(-0.32, 1., r"$(\mathrm{c})$", transform = ax.transAxes, fontsize = 8)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("ThesisPlots/ThesisSurfaceDispersion.png")

