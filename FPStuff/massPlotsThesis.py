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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import handleIntegralData

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

def plotEffectiveMassesComparison():

    filenameFP =  "delMassFP"
    cutoff, dArr, massFPArr = handleIntegralData.retrieveMassData(filenameFP)

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPhP = "massThesis1ForPlotting" + wLOStr + wTOStr
    cutoff, zArr, massSPhPArr = handleIntegralData.readMassesSPhP(filenameSPhP)

    print(np.amin(zArr))

    print(zArr.shape)

    fig = plt.figure(figsize=(6.5, 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1],
                           wspace=0.5, hspace=1., top=0.87, bottom=0.24, left=0.05, right=0.98)
    axFP = plt.subplot(gs[0, 1])
    axSPhP = plt.subplot(gs[0, 3])

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12

    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axFP.plot(dArr, massFPArr * 1e7, color = cmapPink(.45), lw = 0.8)

    #print(dArr[-1])
    #print([massFPArr[-1]])
    #dArrRef = 0.001
    #massFPArrRef = -7.137742047012218e-10
    #axFP.plot(dArr, dArr[-1]/dArr * massFPArr[-1] * 1e7, color = 'red', linestyle = '--', lw = .5)
    #axFP.plot(dArr, dArrRef/dArr * np.abs(massFPArrRef) * 1e7, color = 'blue', linestyle = '--', lw = .5)

    axSPhP.plot(zArr / lambda0, massSPhPArr, color = cmapBone(.45), lw = 0.8)
    axSPhP.axhline(0.1, color = "black", lw = 0.3)
    axFP.set_xscale("log")
    axSPhP.set_xscale("log")
    axFP.set_xlim(np.amin(dArr), np.amax(dArr))
    zmin = 1e-4
    zmax = 1e2
    ymin = 0
    ymax = 1
    axSPhP.set_xlim(zmin, zmax)
    axSPhP.set_ylim(ymin, ymax)

    cmapCustom = LinearSegmentedColormap.from_list('', ['white', 'coral'])
    imN = 50
    X = np.zeros((imN, imN), dtype = float)
    #for xPos in np.arange(imN):
    #    X[:, xPos] = 1. - xPos**(.1) / (imN - 1.)**(.1)
    X[:, 0] = 1.
    X[:, 1] = .7
    X[:, 2] = .5
    X[:, 3] = .3
    X[:, 4] = .2
    X[:, 5] = .1
    X[:, 6] = .08
    X[:, 7] = .05
    X[:, 8] = .04
    X[:, 9] = .03
    X[:, 10] = .02
    X[:, 11] = .01

    axSPhP.imshow(X, interpolation='bicubic', cmap=cmapCustom,
              extent=(1e-4, 1e-2, ymin, ymax), alpha=1.)

    arrowSPhP = mpatches.FancyArrowPatch((0.4, .9), (0.1, .9),
                                     mutation_scale=5,
                                     facecolor = 'black',
                                     lw = .3,
                                     edgecolor="black",
                                     transform=axSPhP.transAxes)
    axSPhP.add_patch(arrowSPhP)

    axSPhP.text(0.15, 0.75, r"$\mathrm{Non{-}perturbative}$", fontsize = 8, transform=axSPhP.transAxes)
    axSPhP.text(0.15, 0.62, r"$\mathrm{regime}$", fontsize = 8, transform=axSPhP.transAxes)

    axFP.set_xlabel("$d [\mathrm{m}]$", fontsize = 8)
    axSPhP.set_xlabel("$z [\lambda_0]$", fontsize = 8)

    axFP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad = 0., fontsize = 8)
    axSPhP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad = 0., fontsize = 8)

    axFP.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axFP.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize = 8)

    axFP.set_yticks([0., -5.])
    axFP.set_yticklabels(["$0$", "$-5$"], fontsize = 8)

    axSPhP.set_xticks([1e-2, 1, 1e2])
    axSPhP.set_xticklabels(["$10^{-2}$", "$1$", "$10^{2}$"], fontsize = 8)

    axSPhP.set_yticks([0., 0.1, 1.])
    axSPhP.set_yticklabels(["$0$", "$0.1$", "$1$"], fontsize = 8)

    axFP.text(1.1 * 1e-6, -.8, r"$\times 10^{-7}$", fontsize = 8)

####### Insets
    axinFP = axFP.inset_axes([0.4, 0.25, 0.5, 0.5])
    axinFP.plot(dArr, np.abs(massFPArr), color = cmapPink(.45), lw = 0.8)
    axinFP.plot(dArr, dArr[-1]/dArr * np.abs(massFPArr[-1]), color = 'red', linestyle = '--', dashes=(4, 4), lw = .5, label = r"$\sim \hspace{-1mm} \frac{1}{d}$")

    axinFP.set_xscale("log")
    axinFP.set_yscale("log")
    axinFP.set_xlim(1e-6, 1e-3)
    axinFP.set_ylim(.5 * 1e-9, 20. * 1e-7)
    #axinFP.set_ylabel(r"$|\Delta m| \, [m_{\rm e}]$", labelpad = 0., fontsize = 6)
    axinFP.set_xlabel("$d [\mathrm{m}]$", fontsize = 6, labelpad = -4)
    axinFP.text(0.05, 1.03, r"$|\Delta m| \, [m_{\rm e}]$", transform = axinFP.transAxes, fontsize = 6)
    #axinFP.text(0.5, .9, r"$\sim \frac{1}{d}$", transform = axinFP.transAxes, fontsize = 6)

    axinFP.set_xticks([1e-6, 1e-3])
    axinFP.set_xticklabels(["$10^{-6}$", "$10^{-3}$"], fontsize = 6)
    axinFP.set_yticks([1e-9, 1e-7])
    axinFP.set_yticklabels(["$10^{-9}$", "$10^{-7}$"], fontsize = 6)
    axinFP.tick_params(axis='y', which='major', pad=0)
    axinFP.tick_params(axis='x', which='major', pad=1)

    legend = axinFP.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.05, 1.05), edgecolor='black', ncol=1, handlelength=2., handletextpad=0.)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

#    axinSPhP = axSPhP.inset_axes([0.35, 0.25, 0.6, 0.6])
#
#    axinSPhP.plot(zArr / lambda0, np.abs(massSPhPArr), color = cmapBone(.45), lw = 0.8)
#    axinSPhP.plot(zArr / lambda0, zArr[-1]**3/zArr**3 * np.abs(massSPhPArr[-1]), color = 'red', linestyle = '--', dashes=(4, 4), lw = .5, label = r"$\sim \hspace{-1mm} \frac{1}{z^3}$")
#    axinSPhP.set_xscale("log")
#    axinSPhP.set_yscale("log")
#    axinSPhP.set_xlim(1e-4, 1e1)
#    axinSPhP.set_ylim(1e-10, 1.)
#
#    axinSPhP.set_xticks([1e-3, 1e0])
#    axinSPhP.set_xticklabels(["$10^{-3}$", "$1$"], fontsize = 6)
#    axinSPhP.set_yticks([1e-6, 1.])
#    axinSPhP.set_yticklabels(["$10^{-6}$", "$1$"], fontsize = 6)
#
#    axinSPhP.tick_params(axis='y', which='major', pad=0)
#    axinSPhP.tick_params(axis='x', which='major', pad=1)
#
#    axinSPhP.set_xlabel("$z [\lambda_0]$", fontsize = 6, labelpad=0)
#    axinSPhP.text(0.05, 1.03, r"$|\Delta m| \, [m_{\rm e}]$", transform = axinSPhP.transAxes, fontsize = 6)
#
#    legend = axinSPhP.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.05, 1.05), edgecolor='black', ncol=1, handlelength=2., handletextpad=0.)
#    legend.get_frame().set_alpha(0.)
#    legend.get_frame().set_boxstyle('Square', pad=0.1)
#    legend.get_frame().set_linewidth(0.0)

####### Dispersion
    axFPDisp = plt.subplot(gs[0, 0])
    axSPhPDisp = plt.subplot(gs[0, 2])

    axFPDisp.text(1., 1.07, r"$\mathrm{Fabry-Perot}$", transform = axFPDisp.transAxes, fontsize = 8)
    axSPhPDisp.text(1., 1.07, r"$\mathrm{Surface}$", transform = axSPhPDisp.transAxes, fontsize = 8)

    axFPDisp.text(.5, .8, r"$\mathrm{increased}$", transform = axFPDisp.transAxes, fontsize = 8, ha='center')
    axFPDisp.text(.5, .68, r"$\mathrm{slope}$", transform = axFPDisp.transAxes, fontsize = 8, ha='center')
    axSPhPDisp.text(.5, .8, r"$\mathrm{decreased}$", transform = axSPhPDisp.transAxes, fontsize = 8, ha='center')
    axSPhPDisp.text(.5, .68, r"$\mathrm{slope}$", transform = axSPhPDisp.transAxes, fontsize = 8, ha='center')

    xArr = np.linspace(-1., 1., 1000)
    yArr = xArr * xArr
    yArrFP = 1.1 * xArr * xArr
    yArrSPhP = 0.6 * xArr * xArr

    axFPDisp.plot(xArr, yArr, color = "gray", lw = 0.5, linestyle = '--', label = r"$\varepsilon_0(k)$")
    axFPDisp.plot(xArr, yArrFP, color = cmapPink(.45) , lw = 0.8, label = r"$\varepsilon(k)$")

    axSPhPDisp.plot(xArr, yArr, color = "gray", lw = 0.5, linestyle = '--', label = r"$\varepsilon_0(k)$")
    axSPhPDisp.plot(xArr, yArrSPhP, color = cmapBone(.45) , lw = 0.8, label = r"$\varepsilon(k)$")

    axFPDisp.set_xlim(-1, 1)
    axFPDisp.set_ylim(0, 1)
    axSPhPDisp.set_xlim(-1, 1)
    axSPhPDisp.set_ylim(0, 1)

    axFPDisp.set_xlabel("$k \, [\mathrm{arb. \, units}]$")
    axSPhPDisp.set_xlabel("$k \, [\mathrm{arb. \, units}]$")

    axFPDisp.set_ylabel(r"$\varepsilon(k) \, [\mathrm{arb. \, units}]$")
    axSPhPDisp.set_ylabel(r"$\varepsilon(k) \, [\mathrm{arb. \, units}]$")

    axFPDisp.set_xticks([])
    axFPDisp.set_yticks([])

    axSPhPDisp.set_xticks([])
    axSPhPDisp.set_yticks([])

    arrowFP1 = mpatches.FancyArrowPatch((0.1, 0.45), (0.11, .9),
                                     mutation_scale=5,
                                     facecolor = cmapPink(.45),
                                     lw = .3,
                                     edgecolor="black",
                                     transform=axFPDisp.transAxes)

    arrowFP2 = mpatches.FancyArrowPatch((0.9, 0.45), (0.89, .9),
                                     mutation_scale=5,
                                     facecolor = cmapPink(.45),
                                     lw = .3,
                                     edgecolor="black",
                                     transform=axFPDisp.transAxes)
    axFPDisp.add_patch(arrowFP1)
    axFPDisp.add_patch(arrowFP2)

    arrowSPhP1 = mpatches.FancyArrowPatch((0.1, 0.8), (0.11, .2),
                                     mutation_scale=5,
                                     facecolor = cmapBone(.45),
                                     lw = .3,
                                     edgecolor="black",
                                     transform=axSPhPDisp.transAxes)

    arrowSPhP2 = mpatches.FancyArrowPatch((0.9, 0.8), (0.89, .2),
                                     mutation_scale=5,
                                     facecolor = cmapBone(.45),
                                     lw = .3,
                                     edgecolor="black",
                                     transform=axSPhPDisp.transAxes)
    axSPhPDisp.add_patch(arrowSPhP1)
    axSPhPDisp.add_patch(arrowSPhP2)

    legend = axFPDisp.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.05, .93), edgecolor='black', ncol=2, handlelength=1., handletextpad=.5, columnspacing=1.0)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(0.0)

    legend = axSPhPDisp.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.05, .93), edgecolor='black', ncol=2, handlelength=1., handletextpad=.5, columnspacing=1.0)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axFP.spines[axis].set_linewidth(.5)
        axinFP.spines[axis].set_linewidth(.5)
        axSPhP.spines[axis].set_linewidth(.5)
#        axinSPhP.spines[axis].set_linewidth(.5)
        axFPDisp.spines[axis].set_linewidth(.5)
        axSPhPDisp.spines[axis].set_linewidth(.5)


    plt.savefig("FPPlotsSaved/EffectiveMassesThesis1.png")



def plotEffectiveHoppingComparison():

    filenameFP =  "delHoppingFP"
    cutoff, dArr, massFPArr = handleIntegralData.retrieveMassData(filenameFP)

    filenameSPhP = "hoppingThesis1ForPlotting"
    cutoff, zArr, massSPhPArr = handleIntegralData.readMassesSPhP(filenameSPhP)

    fig = plt.figure(figsize=(5., 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1],
                           wspace=0.4, hspace=0.6, top=0.75, bottom=0.25, left=0.1, right=0.98)
    axFP = plt.subplot(gs[0, 0])
    axSPhP = plt.subplot(gs[0, 1])
    axSPhPUpperAx = axSPhP.twiny()

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axFP.plot(dArr, -massFPArr * 1e12, color = cmapPink(.5), lw = 0.8)


    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf

    axSPhP.plot(zArr / lambda0, -massSPhPArr * 1e2, color = cmapBone(.5), lw = 0.8)
    axSPhPUpperAx.plot(zArr, -massSPhPArr * 1e2, color = cmapBone(.5), lw = 0.8)

    axFP.set_xscale("log")
    axSPhP.set_xscale("log")
    axSPhPUpperAx.set_xscale("log")

    axFP.set_xlim(np.amin(dArr), np.amax(dArr))
    axFP.set_ylim(0., 4.)
    axSPhP.set_xlim(1e-5, 1e2)
    axSPhP.set_xlim(np.amin(zArr[1:] / lambda0), np.amax(zArr[1:] / lambda0))
    axSPhP.set_ylim(-2.2, 0.)
    axSPhPUpperAx.set_xlim(np.amin(zArr[1:]), np.amax(zArr[1:]))
    axSPhPUpperAx.set_ylim(-2.2, 0.)
    #axSPhPUpperAx.xaxis.set_ticks_position('top')

    axFP.set_xlabel("$d [\mathrm{m}]$", fontsize = 8)
    axSPhP.set_xlabel("$z_0 [\lambda_0]$", fontsize = 8)
    axSPhPUpperAx.set_xlabel("$z_0 [\mathrm{m}]$", fontsize = 8)

    axFP.set_ylabel(r"$\frac{\Delta t}{t}[\%]$", labelpad = 2., fontsize = 8)
    axSPhP.set_ylabel(r"$\frac{\Delta t}{t}[\%]$", labelpad = 2., fontsize = 8)

    axFP.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axFP.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize = 8)

    axFP.set_yticks([0., 5.])
    axFP.set_yticklabels(["$0$", "$5$"], fontsize = 8)

    axSPhP.set_xticks([1e-4, 1e-2, 1e0, 1e2])
    axSPhP.set_xticklabels(["$10^{-4}$", "$10^{-2}$", "$1$", "$10^{2}$"], fontsize = 8)
    axSPhPUpperAx.set_xticks([1e-9, 1e-6, 1e-3])
    axSPhPUpperAx.set_xticklabels(["$10^{-9}$", "$10^{-6}$", "$10^{-3}$"], fontsize = 8)

    axSPhP.set_yticks([-2, -1, 0.])
    axSPhP.set_yticklabels(["$-2$", "$-1$", "$0$"], fontsize = 8)

    axFP.text(.02, .85, r"$\times 10^{-10}$", fontsize = 8, transform = axFP.transAxes)
    axFP.text(.3, .5, r"$\mathrm{Fabry-Perot}$", transform = axFP.transAxes, fontsize = 8)
    axSPhP.text(.35, .5, r"$\mathrm{Surface}$", transform = axSPhP.transAxes, fontsize = 8)

    axFP.text(-0.185, 1.35, r"$\mathrm{(a)}$", transform = axFP.transAxes, fontsize = 8)
    axSPhP.text(-0.24, 1.35, r"$\mathrm{(b)}$", transform = axSPhP.transAxes, fontsize = 8)

    for axis in ['top', 'bottom', 'left', 'right']:
        axFP.spines[axis].set_linewidth(.5)
        axSPhP.spines[axis].set_linewidth(.5)
        axSPhPUpperAx.spines[axis].set_linewidth(.5)


    plt.savefig("FPPlotsSaved/EffectiveHoppingThesis1.png")
