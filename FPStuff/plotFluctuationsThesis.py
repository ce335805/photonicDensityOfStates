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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import handleIntegralData

import scipy.constants as consts

fontsize = 8

mpl.rcParams['font.family'] = 'serif'
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

def plotFluctuationsAandENaturalUnits(dArr, flucE, flucA, freqArr, cutoff):
    fig = plt.figure(figsize=(6.5 / 3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1, wspace=0.35, hspace=0., top=0.9, bottom=0.2, left=0.28, right=0.72)
    axE = plt.subplot(gs[0, 0])
    axA = axE.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    unitFacE = consts.epsilon_0 / (consts.hbar * freqArr) * 1e-18
    unitFacA = consts.epsilon_0 * freqArr / consts.hbar * 1e-18

    axE.plot(dArr, flucE * unitFacE, color=cmapBone(.45), linestyle='', marker='x', markersize=2.)
    axA.plot(dArr, flucA * unitFacA * 1e-24, color=cmapPink(.45), linestyle='', marker='x', markersize=2.)
    axE.set_ylim(1e-8, 0.15)
    axA.set_ylim(-0.0022, .015)
    axE.set_xlim(np.amin(dArr), np.amax(dArr))
    axA.set_xlim(np.amin(dArr), np.amax(dArr))
    axE.set_xscale('log')
    axE.set_yscale('log')
    axA.set_xscale('log')

    axE.set_xticks([1e-6, 1e-4])
    axE.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$"], fontsize = 8)

    axE.set_yticks([1e-6, 1e-4, 1e-2])
    axE.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"], fontsize = 8)

    axA.set_yticks([0., 0.01])
    axA.set_yticklabels([r"$0$", r"$0.01$"])

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    axE.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$", fontsize=8, labelpad = 2)
    axE.set_ylabel(r"$\langle \widehat{E\,}_{\! ||}^{\! 2} \rangle_{\rm eff} \left[ \frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \, \omega_0}{\varepsilon_0 \mu \mathrm{m}^3} \right]$", fontsize=8,
                   labelpad=2)
    axA.set_ylabel(
        r"$\langle \widehat{A\,}^{\! 2} \rangle_{\rm eff} \left[ \frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt}}{\varepsilon_0 \mu \mathrm{m}^3} \right]$",
        fontsize=8, labelpad=4)


    plt.arrow(0.78, 0.12, 0.2, 0.2, transform=axA.transAxes, length_includes_head=True, color=cmapPink(.45), head_width = 0.05, head_length = 0.05)
    plt.arrow(0.25, 0.7, -0.23, -0.15, transform=axA.transAxes, length_includes_head=True, color=cmapBone(.45), head_width = 0.05, head_length = 0.05)

    axA.text(2. * 1e-6, -0.0015, r"$\to 0 \, (\Lambda \to \infty)$", fontsize = 6, color = cmapPink(.45))

    # legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    axE.text(-0.6, 1.07, r"$\mathrm{(a)}$", fontsize = 8, transform = axE.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)
        axA.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/ThesisFlucEandANatUnits.png")



def plotFluctuationsEExpUnits(dArr, flucE, freqArr, cutoff):
    fig = plt.figure(figsize=(2.2, 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 1, wspace=0.35, hspace=0., top=0.95, bottom=0.25, left=0.28, right=0.82)

    freqArr = np.pi * consts.c / dArr
    freqArrInEv = consts.hbar * np.pi * consts.c / dArr / consts.e

    axE = plt.subplot(gs[0, 0])
    axFreq = axE.twinx()

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    axE.plot(dArr, np.sqrt(flucE) * 1e-6, color=cmapBone(.45), linestyle='', marker='x', markersize=2.)
    axFreq.plot(dArr, freqArrInEv, color = 'red', lw = .4)
    axE.set_ylim(0, 0.04)
    axFreq.set_ylim(0, 0.65)
    axE.set_xlim(np.amin(dArr), np.amax(dArr))
    axE.set_xscale('log')
    #axE.set_yscale('log')

    axE.set_xticks([1e-6, 1e-4])
    axE.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$"], fontsize = 8)

    #axE.set_yticks([1e-6, 1e-4, 1e-2])
    #axE.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"], fontsize = 8)

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    axE.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$", fontsize=8, labelpad = 2)
    axE.set_ylabel(r"$\sqrt{\langle \widehat{E\,}_{\! ||}^{\! 2} \rangle_{\rm eff}} \left[ \frac{\mathrm{MV}}{\mathrm{m}} \right]$", fontsize=8,
                   labelpad=4)

    axFreq.set_ylabel(r"$h \hspace{-1.3mm} \rule[1.4ex]{0.3em}{0.4pt} \, \omega_0 \, \left[ \mathrm{eV} \right]$", fontsize=8,
                   labelpad=4)
    # legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    axE.text(.5, .5, r"$\mathrm{Fabry-Perot}$", fontsize = 8, transform = axE.transAxes, ha='center', va='center')

    axE.text(-0.5, 0.98, r"$\mathrm{(a)}$", fontsize = 8, transform = axE.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)
        axFreq.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/ThesisFlucEExpUnits.png")


