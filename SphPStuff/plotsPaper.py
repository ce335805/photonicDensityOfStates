import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib.colors import LinearSegmentedColormap
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

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import epsilonFunctions as epsFunc
import dosAsOfFreq

import produceFreqDataV2 as prodV2

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

def retrieveMassData(fileName):
    dir = "../SphPStuff/savedData/PaperData/"
    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'r')
    cutoff = h5f['cutoff'][:]
    dArr = h5f['distance'][:]
    massArr = h5f['massArr'][:]
    h5f.close()

    return (cutoff[0], dArr, massArr)

def readMassesSPhP(filename):

    dir = "../SphPStuff/savedData/PaperData/"
    fileName = dir + filename + ".hdf5"

    h5f = h5py.File(fileName, 'r')
    cutoff = h5f['cutoff'][:]
    mArr = h5f['delM'][:]
    zArr = h5f['zArr'][:]
    h5f.close()
    return (cutoff, zArr, mArr)

def plotEffectiveMassesComparison():

    filenameFP =  "delMassFPPaperNew"
    cutoff, dArr, massFPArr = retrieveMassData(filenameFP)

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    #filenameSPhP = "massThesis1ForPlotting" + wLOStr + wTOStr + "Thesis"
    filenameSPhP = "massForPaperNew" + wLOStr + wTOStr
    cutoff, zArr, massSPhPArr = readMassesSPhP(filenameSPhP)

    fig = plt.figure(figsize=(7., 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 3, 3],
                           wspace=0.4, hspace=1., top=0.8, bottom=0.24, left=0.03, right=0.98)
    axFP = plt.subplot(gs[0, 1])
    axSPhP = plt.subplot(gs[0, 2])
    axSPhPUpperAx = axSPhP.twiny()


    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12

    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axFP.plot(dArr, massFPArr * 1e9, color = cmapBone(.45), lw = 1.2)

    axSPhP.plot(zArr / lambda0, massSPhPArr, color = cmapPink(.52), lw = 1.2)
    axSPhPUpperAx.plot(zArr, massSPhPArr, color = cmapPink(.52), lw = 1.2)
    #axSPhP.axhline(0.1, color = "black", lw = 0.3)
    axFP.set_xscale("log")
    axSPhP.set_xscale("log")
    axSPhPUpperAx.set_xscale("log")
    axFP.set_xlim(np.amin(dArr), np.amax(dArr))
    zmin = 0.7 * 1e-4
    zmax = 2.5 * 1e-2
    ymin = 0
    ymax = 0.3
    axSPhP.set_xlim(zmin, zmax)
    axSPhP.set_ylim(ymin, ymax)
    axSPhPUpperAx.set_ylim(ymin, ymax)
    axSPhPUpperAx.set_xlim(zmin * lambda0, zmax * lambda0)
    axFP.set_ylim(-2., 0.)

    cmapCustom = LinearSegmentedColormap.from_list('', ['white', cmapBone(0.65)])
    imN = 50
    X = np.zeros((imN, imN), dtype=float)
    # for xPos in np.arange(imN):
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
    X[:, 12] = .005
    X[:, 13] = .0025
    X[:, 14] = .001
    X[:, 15] = .0005
    X[:, 16] = .00025
    X[:, 17] = .0001

    axSPhP.imshow(X, interpolation='bicubic', cmap=cmapCustom,
                  extent=(zmin, 3 * 1e-3, ymin, ymax), alpha=1.)

    axFP.set_xlabel("$d [\mathrm{m}]$", fontsize = 8)
    axSPhP.set_xlabel("$z [\lambda_0]$", fontsize = 8)
    #axSPhPUpperAx.set_xlabel("$z [\mu \mathrm{m}]$", fontsize = 8, labelpad = 0)

    axFP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad = 0., fontsize = 8)
    axSPhP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad = 0., fontsize = 8)

    axFP.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axFP.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize = 8)

    axFP.set_yticks([0., -1.])
    axFP.set_yticklabels(["$0$", "$-1$"], fontsize = 8)

    axSPhP.set_xticks([1e-4, 1e-2])
    axSPhP.set_xticklabels(["$10^{-4}$", "$10^{-2}$"], fontsize = 8)

    axSPhPUpperAx.set_xticks([1e-8, 1e-6])
    axSPhPUpperAx.set_xticklabels(["$10 \mathrm{nm}$", "$1 \mu \mathrm{m}$"], fontsize = 8)

    axSPhP.set_yticks([0., 0.3])
    axSPhP.set_yticklabels(["$0$", "$0.3$"], fontsize = 8)

    axFP.text(1.1 * 1e-6, -.3, r"$\times 10^{-9}$", fontsize = 8)

####### Insets
    axinFP = axFP.inset_axes([0.4, 0.25, 0.5, 0.55])
    axinFP.plot(dArr, np.abs(massFPArr), color = cmapBone(.45), lw = 1.6)
    axinFP.plot(dArr, dArr[-1]/dArr * np.abs(massFPArr[-1]), color = cmapPink(.55), linestyle = '--', dashes=(4, 4), lw = 1.2, label = r"$\sim \hspace{-0.5mm} d^{-1}$")

    axinFP.set_xscale("log")
    axinFP.set_yscale("log")
    axinFP.set_xlim(1e-6, 1e-3)
    axinFP.set_ylim(1e-12, 1e-8)
    axinFP.set_ylabel(r"$ \log \left(|\Delta m|\right)$", labelpad = 2, fontsize = 7)
    axinFP.set_xlabel("$ \log(d)$", fontsize = 7, labelpad = 2)
    #axinFP.text(0.05, 1.03, r"$|\Delta m| \, [m_{\rm e}]$", transform = axinFP.transAxes, fontsize = 6)
    #axinFP.text(0.5, .9, r"$\sim \frac{1}{d}$", transform = axinFP.transAxes, fontsize = 6)

    axinFP.set_xticks([])
    #axinFP.set_xticks([1e-6, 1e-3])
    #axinFP.set_xticklabels(["$10^{-6}$", "$10^{-3}$"], fontsize = 6)
    axinFP.set_yticks([])
    #axinFP.set_yticks([1e-11, 1e-9])
    #axinFP.set_yticklabels(["$10^{-11}$", "$10^{-9}$"], fontsize = 6)
    axinFP.tick_params(axis='y', which='major', pad=0)
    axinFP.tick_params(axis='x', which='major', pad=1)

    legend = axinFP.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.1, 1.1), edgecolor='black', ncol=1, handlelength=2, handletextpad=0.5)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

#    axinSPhP = axSPhP.inset_axes([0.35, 0.25, 0.6, 0.6])

#    axinSPhP.plot(zArr / lambda0, np.abs(massSPhPArr), color = cmapBone(.45), lw = 0.8)
#    axinSPhP.plot(zArr / lambda0, zArr[-1]**3/zArr**3 * np.abs(massSPhPArr[-1]), color = 'red', linestyle = '--', dashes=(4, 4), lw = .5, label = r"$\sim \hspace{-1mm} \frac{1}{z^3}$")
#    axinSPhP.set_xscale("log")
#    axinSPhP.set_yscale("log")
#    axinSPhP.set_xlim(1e-2, 1e1)
#    axinSPhP.set_ylim(1e-9, 1e-5)
#
#    axinSPhP.set_xticks([1e-1, 1e1])
#    axinSPhP.set_xticklabels(["$10^{-1}$", "$10^{1}$"], fontsize = 6)
#    axinSPhP.set_yticks([1e-8, 1e-6])
#    axinSPhP.set_yticklabels(["$10^{-8}$", "$10^{-6}$"], fontsize = 6)

#    axinSPhP.tick_params(axis='y', which='major', pad=0)
#    axinSPhP.tick_params(axis='x', which='major', pad=1)
#
#    axinSPhP.set_xlabel("$z [\lambda_0]$", fontsize = 6, labelpad=0)
#    axinSPhP.text(0.05, 1.03, r"$|\Delta m| \, [m_{\rm e}]$", transform = axinSPhP.transAxes, fontsize = 6)

#    legend = axinSPhP.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.05, 1.05), edgecolor='black', ncol=1, handlelength=2., handletextpad=0.)
#    legend.get_frame().set_alpha(0.)
#    legend.get_frame().set_boxstyle('Square', pad=0.1)
#    legend.get_frame().set_linewidth(0.0)

    axIm = plt.subplot(gs[0, 0])
    axIm.set_xticks([])
    axIm.set_yticks([])
    arr_img = mpimg.imread("../SphPStuff/PaperPlots/HeadComic.png")
    im = OffsetImage(arr_img, zoom=.12)
    ab = AnnotationBbox(im, (0.05, -0.32), xycoords='axes fraction', box_alignment=(0, 0), frameon=False, pad=0,
                        boxcoords="offset points")
    axIm.add_artist(ab)

    axFP.text(.2, 1.07, r"$\mathrm{Fabry {-} Perot \, Cavity}$", fontsize=8, transform=axFP.transAxes, zorder = 666)
    axSPhP.text(.1, .7, r"$\mathrm{Surface \, Phonon {-} Polaritons}$", fontsize=8, transform=axSPhP.transAxes, zorder = 666)


    axIm.text(-.05, 1.07, r"$\mathrm{(a)}$", fontsize=8, transform=axIm.transAxes, zorder = 666)
    axFP.text(-.22, 1.07, r"$\mathrm{(b)}$", transform = axFP.transAxes, fontsize = 8)
    axSPhP.text(-.22, 1.07, r"$\mathrm{(c)}$", transform = axSPhP.transAxes, fontsize = 8)

    for axis in ['top', 'bottom', 'left', 'right']:
        axFP.spines[axis].set_linewidth(.5)
        axinFP.spines[axis].set_linewidth(.5)
        axSPhP.spines[axis].set_linewidth(.5)
        axSPhPUpperAx.spines[axis].set_linewidth(.5)
#        axinSPhP.spines[axis].set_linewidth(.5)
        axIm.spines[axis].set_linewidth(.0)

    plt.savefig("../SphPStuff/PaperPlots/Masses.png")

def EffectiveMassSPhP():

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    #filenameSPhP = "massThesis1ForPlotting" + wLOStr + wTOStr + "Thesis"
    filenameSPhP = "massForPaperNew" + wLOStr + wTOStr
    cutoff, zArr, massSPhPArr = readMassesSPhP(filenameSPhP)

    fig = plt.figure(figsize=(7. / 3.4, 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.0, hspace=0., top=0.85, bottom=0.24, left=0.19, right=0.98)
    axSPhP = plt.subplot(gs[0, 0])
    axSPhPUpperAx = axSPhP.twiny()

    wInf = np.sqrt(wLO ** 2 + wTO ** 2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    axSPhP.plot(zArr / lambda0, massSPhPArr, color=cmapPink(.43), lw=1.2)
    axSPhPUpperAx.plot(zArr, massSPhPArr, color=cmapPink(.43), lw=1.2)
    axSPhP.axhline(0.1, color = "black", lw = 0.3)
    axSPhP.set_xscale("log")
    axSPhPUpperAx.set_xscale("log")
    zmin = 0.7 * 1e-4
    zmax = 2.5 * 1e-2
    ymin = -0.0025
    ymax = 0.3
    axSPhP.set_xlim(zmin, zmax)
    axSPhP.set_ylim(ymin, ymax)
    axSPhPUpperAx.set_ylim(ymin, ymax)
    axSPhPUpperAx.set_xlim(zmin * lambda0, zmax * lambda0)

    cmapCustom = LinearSegmentedColormap.from_list('', ['white', '#bcd4e3'])
    imN = 50
    X = np.zeros((imN, imN), dtype=float)
    # for xPos in np.arange(imN):
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
    X[:, 12] = .005
    X[:, 13] = .0025
    X[:, 14] = .001
    X[:, 15] = .0005
    X[:, 16] = .00025
    X[:, 17] = .0001

    axSPhP.imshow(X, interpolation='bicubic', cmap=cmapCustom,
                  extent=(zmin, 2 * 1e-3, ymin, ymax), alpha=1.)

    axSPhP.set_xlabel("$z [\lambda_0]$", fontsize=8)
    # axSPhPUpperAx.set_xlabel("$z [\mu \mathrm{m}]$", fontsize = 8, labelpad = 0)

    axSPhP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad=0., fontsize=8)

    axSPhP.set_xticks([1e-4, 1e-2])
    axSPhP.set_xticklabels(["$10^{-4}$", "$10^{-2}$"], fontsize=8)

    axSPhPUpperAx.set_xticks([1e-8, 1e-6])
    axSPhPUpperAx.set_xticklabels(["$10 \mathrm{nm}$", "$1 \mu \mathrm{m}$"], fontsize=8)

    axSPhP.set_yticks([0., 0.1, 0.3])
    axSPhP.set_yticklabels(["$0$", "$0.1$", "$0.3$"], fontsize=8)

    axSPhP.text(.3, .7, r"$\mathrm{Surface \, Polaritons}$", fontsize=8, transform=axSPhP.transAxes,
                zorder=666)

    axSPhP.text(.85, .36, r"$\mathrm{Ultra-Strong Coupling}$", fontsize=7, transform=axSPhP.transAxes,
                zorder=666)


    axSPhP.text(-.23, 1.12, r"$\mathrm{(a)}$", transform=axSPhP.transAxes, fontsize=8)

    for axis in ['top', 'bottom', 'left', 'right']:
        axSPhP.spines[axis].set_linewidth(.5)
        axSPhPUpperAx.spines[axis].set_linewidth(.5)

    plt.savefig("../SphPStuff/PaperPlots/MassesSPhP.png")

def EffectiveMassFP():
    filenameFP = "delMassFPPaperNew"
    cutoff, dArr, massFPArr = retrieveMassData(filenameFP)

    fig = plt.figure(figsize=(7. / 3., 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.0, hspace=0., top=0.85, bottom=0.24, left=0.18, right=0.93)
    axFP = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axFP.plot(dArr, massFPArr * 1e9, color=cmapBone(.45), lw=1.2)

    axFP.set_xscale("log")
    axFP.set_xlim(np.amin(dArr), np.amax(dArr))

    axFP.set_ylim(-2., 0.)
    axFP.set_xlabel("$d [\mathrm{m}]$", fontsize=8)
    axFP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad=0., fontsize=8)

    axFP.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axFP.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize=8)

    axFP.set_yticks([0., -1.])
    axFP.set_yticklabels(["$0$", "$-1$"], fontsize=8)

    axFP.text(1.1 * 1e-6, -.3, r"$\times 10^{-9}$", fontsize=8)

    ####### Insets
    axinFP = axFP.inset_axes([0.4, 0.25, 0.5, 0.55])
    axinFP.plot(dArr, np.abs(massFPArr), color=cmapBone(.45), lw=1.6)
    axinFP.plot(dArr, dArr[-1] / dArr * np.abs(massFPArr[-1]), color=cmapPink(.55), linestyle='--', dashes=(4, 4),
                lw=1.2, label=r"$\sim \hspace{-0.5mm} d^{-1}$")

    axinFP.set_xscale("log")
    axinFP.set_yscale("log")
    axinFP.set_xlim(1e-6, 1e-3)
    axinFP.set_ylim(1e-12, 1e-8)
    axinFP.set_ylabel(r"$ \log \left(|\Delta m|\right)$", labelpad=2, fontsize=7)
    axinFP.set_xlabel("$ \log(d)$", fontsize=7, labelpad=2)

    axinFP.set_xticks([])
    axinFP.set_yticks([])
    axinFP.tick_params(axis='y', which='major', pad=0)
    axinFP.tick_params(axis='x', which='major', pad=1)

    axFP.text(.2, 1.07, r"$\mathrm{Fabry {-} Perot \, Cavity}$", fontsize=8, transform=axFP.transAxes, zorder = 666)
    axFP.text(-.22, 1.12, r"$\mathrm{(c)}$", transform = axFP.transAxes, fontsize = 8)

    legend = axinFP.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.1, 1.1), edgecolor='black', ncol=1,
                           handlelength=2, handletextpad=0.5)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axFP.spines[axis].set_linewidth(.5)
        axinFP.spines[axis].set_linewidth(.5)

    plt.savefig("../SphPStuff/PaperPlots/MassesFP.png")

def plotClassicalAnalogy():

    wLO = 32.04 * 1e12
    wTO = 1e6
    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    #filenameSPhP = "./savedData/PaperData/massThesisFinal" + wLOStr + wTOStr + ".hdf5"
    #filenameSPhP = "./savedData/PaperData/massForPaperNew" + wLOStr + wTOStr + ".hdf5"
    filenameSPhP = "./savedData/PaperData/massForPaperLowerCutoff10GHz" + wLOStr + wTOStr + ".hdf5"

    cutoff, zArrSPP, massSPPArr = prodV2.readMasses(filenameSPhP)
    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf
    unitFac = consts.m_e / (consts.hbar * wLO / consts.c**2)

    fig = plt.figure(figsize=(3.3, 3.), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 3],
                           wspace=0.4, hspace=0.4, top=0.9, bottom=0.15, left=0.2, right=0.95)

    axIm = plt.subplot(gs[0, 0])
    axZoom = plt.subplot(gs[1, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axZoom.plot(zArrSPP / lambda0, np.abs(massSPPArr) * unitFac, color = cmapBone(.25), lw = 1.2, label = r"$\omega_{\rm p} = 32 \mathrm{THz}$")

    axZoom.plot(zArrSPP / lambda0, np.abs(massSPPArr[-1]) * zArrSPP[-1]**3 / zArrSPP**3 * unitFac, color = "red", lw = .4)
    fitInd = 150
    axZoom.plot(zArrSPP / lambda0, np.abs(massSPPArr[fitInd]) * zArrSPP[fitInd]**1 / zArrSPP**1 * unitFac, color = "red", lw = .4)

    axZoom.set_xscale("log")
    axZoom.set_yscale("log")

    axZoom.set_xlim(1e-2, 1e1)
    axZoom.set_ylim(1e-5, 2. * 1e1)

    axZoom.set_xlabel("$z [\lambda_0]$", fontsize = 8)
    axZoom.set_ylabel(r"$|\Delta m| \, \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \omega_{\rm p}}{c^2} \right]$", labelpad = 4., fontsize = 8)

    axZoom.text(0.05, 0.8, r"$\sim z^{-3}$", color = 'red', transform = axZoom.transAxes)
    axZoom.text(0.02, 0.32, r"$\sim z^{-1}$", color = 'red', transform = axZoom.transAxes)

    axZoom.text(0.1, 1.03, r"$\mathrm{Near{-}Field}$", transform = axZoom.transAxes, fontsize = 8)
    axZoom.text(0.65, 1.03, r"$\mathrm{Far{-}Field}$", transform = axZoom.transAxes, fontsize = 8)

##################################
    # Inset
##################################

    axIn = axZoom.inset_axes([0.49, 0.48, 0.5, 0.5])
    axIn.plot(zArrSPP / lambda0, massSPPArr * unitFac * 1e4, color = cmapBone(.25), lw = 1.2, label = r"$\omega_{\rm p} = 32 \mathrm{THz}$")
    axIn.axhline(0., lw = 0.5, color = 'black')
    axIn.set_xlim(1e-2, 6)
    axIn.set_ylim(- 2.8, 2.8)
    axIn.set_xlabel("$z [\lambda_0]$", fontsize = 7, labelpad = 0)
    axIn.set_ylabel(r"$\Delta m \, \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \omega_{\rm p}}{c^2} \right]$", labelpad = 0., fontsize = 7)

    axIn.set_yticks([-2., 0, 2.])
    axIn.set_yticklabels([r"$-2$", r"$0$", r"$2$"], fontsize = 7)
    axIn.set_xticks([0, 1, 5])
    axIn.set_xticklabels([r"$0$", r"$1$", r"$5$"], fontsize = 7)

#########################################################

    #legend = axZoom.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    axIm.set_xticks([])
    axIm.set_yticks([])
    arr_img = mpimg.imread('./PaperPlots/pngMassComic.png')
    im = OffsetImage(arr_img, zoom = .27)
    ab = AnnotationBbox(im, (0.1, -0.2), xycoords='axes fraction', box_alignment=(0, 0), frameon=False, pad=0, boxcoords="offset points")
    axIm.add_artist(ab)

    axIm.text(-0., 1.04, r"$\mathrm{(a)}$", fontsize = 8, transform = axIm.transAxes)
    axZoom.text(-0., 1.04, r"$\mathrm{(b)}$", fontsize = 8, transform = axZoom.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axZoom.spines[axis].set_linewidth(.5)
        #axIn.spines[axis].set_linewidth(.5)
        axIm.spines[axis].set_linewidth(0.)


    plt.savefig("./PaperPlots/ClassicalAnalogy.png")

def createDosPlot():

    wArrSubdivisions = 100

    epsInf = 1.
    #numbers for sto
    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    L = 1.
    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), 200, endpoint=True, base = 10)
    zArr = np.append([L / 4.], zArr)

    wArrSPhP, dosSPhPBulk, dosSPhPSurf = getDosDataSPP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

    dFP, wArrFP, dosFP = getDosFP()

    #wTO = 1e6

    zArrM, mArr, mBulkArr = getMass(wLO, wTO)

    plotDos(zArr, wArrSPhP, dosSPhPBulk + dosSPhPSurf, dosSPhPSurf, dFP, wArrFP, dosFP, zArrM, mArr, mBulkArr, wLO, wTO)

def getDosFP():

    dir = "./savedData/PaperData/"

    fileName = dir + "DosFP.h5"
    h5f = h5py.File(fileName, 'r')
    d = h5f['d'][:]
    wArr = h5f['wArr'][:]
    dos = h5f['dos'][:]
    h5f.close()

    return (d, wArr, dos)


def getDosDataSPP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):

    dosTETotal, dosTMPara = prodV2.retrieveDosPara(
        wArrSubdivisions,
        zArr,
        wLO,
        wTO,
        epsInf,
        L)

    wArrOne = prodV2.defineFreqArrayOne(wArrSubdivisions)
    dosSurf = np.zeros((len(wArrOne), len(zArr)))
    for wInd, wVal in enumerate(wArrOne):
        epsilon = epsFunc.epsilon(wVal, wLO, wTO, epsInf)
        dosSurf[wInd, :] = 1. / (1. + np.abs(epsilon)) * dosAsOfFreq.getDosTMSurf(wVal, zArr, L, wLO, wTO, epsInf)

    return (wArrOne, dosTETotal + dosTMPara, dosSurf)

def getMass(wLO, wTO):

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    #filename = "./savedData/PaperData/massForPaperInclBulk" + wLOStr + wTOStr + ".hdf5"
    filename = "./savedData/PaperData/massForPaperNew" + wLOStr + wTOStr + ".hdf5"
    h5f = h5py.File(filename, 'r')
    mArr = h5f['delM'][:]
    mBulkArr = h5f['delMBulk'][:]
    zArr = h5f['zArr'][:]
    h5f.close()

    return (zArr, mArr, mBulkArr)

def plotDos(zArr, wArrSPhP, dosSPhPBulk, dosSPhPSurf, dFP, wArrFP, dosFP, zArrM, mArr, mBulkArr, wLO, wTO):
    wInf = np.sqrt(wLO ** 2 + wTO ** 2) / np.sqrt(1 + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    fig = plt.figure(figsize=(7., 1.8), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1],
                           wspace=0.35, hspace=0., top=0.85, bottom=0.25, left=0.05, right=0.995)

    axSPhP = plt.subplot(gs[0, 1])
    axFP = plt.subplot(gs[0, 0])
    axM = plt.subplot(gs[0, 2])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    indArr = np.array([1, 30, 60, 92], dtype=int)
    #indArr = np.array([1, 30, 60, 80, 180], dtype=int)

    # axSPhP.plot(wArr, dos[:, indArr[0]], color=cmapBone(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP[0:], dosSPhPBulk[0:, indArr[1]], color=cmapPink(0.1), lw=1.2,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP[0:], dosSPhPBulk[0:, indArr[2]], color=cmapPink(0.4), lw=1.2,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP[0:], dosSPhPBulk[0:, indArr[3]], color=cmapPink(0.6), lw=1.2,
            label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    #
    axSPhP.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    axSPhP.set_xlim(0, 40 * 1e12)
    axSPhP.set_ylim(0., 5.)

    axSPhP.set_xticks([0., wTO, wInf, wLO])
    axSPhP.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize=8)
    axSPhP.set_yticks([0., 2. / 3., 2])
    axSPhP.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axSPhP.set_xlabel(r"$\omega$")
    axSPhP.set_ylabel(r"$\rho_{||} / \rho_0$")

#    axinSPhP = axSPhP.inset_axes([0.55, 0.44, 0.4, 0.5])
#
#    axinSPhP.plot(wArrSPhP, dosSPhPBulk[:, indArr[1]], color=cmapPink(0.1), lw=.7,
#            label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
#    axinSPhP.plot(wArrSPhP, dosSPhPBulk[:, indArr[2]], color=cmapPink(0.4), lw=.7,
#            label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
#    axinSPhP.plot(wArrSPhP, dosSPhPBulk[:, indArr[3]], color=cmapPink(0.6), lw=.7,
#            label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
#
#    axinSPhP.set_xlim(0, 35 * 1e12)
#    axinSPhP.set_ylim(0., 120.)
#
#    axinSPhP.set_xticks([0., wTO, wInf, wLO])
#    axinSPhP.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize=6)
#    axinSPhP.set_yticks([0., 100])
#    axinSPhP.set_yticklabels([r"$0$", r"$100$"], fontsize=6)
#
#    axinSPhP.set_xlabel(r"$\omega$", fontsize = 6, labelpad = -2)
#    axinSPhP.set_ylabel(r"$\rho_{||} / \rho_0$", fontsize = 0)

    axSPhP.text(0.05, 0.8, r"$\mathrm{Dielectric}$", fontsize=8, transform=axSPhP.transAxes)
    axSPhP.text(0.05, 0.68, r"$\mathrm{Behavior}$", fontsize=8, transform=axSPhP.transAxes)
    axSPhP.text(0.65, 0.4, r"$\mathrm{Surface \, Mode}$", fontsize=8, transform=axSPhP.transAxes)

    arrowSPhP = mpatches.FancyArrowPatch((0.75, 0.525), (0.55, 0.7),
                                     mutation_scale=0.1,
                                     facecolor = 'black',
                                     lw = .5,
                                     edgecolor="black",
                                     transform=axSPhP.transAxes)

    axSPhP.add_patch(arrowSPhP)


    axSPhP.text(-0.11, 1.1, r"$(\mathrm{b})$", fontsize=8, transform=axSPhP.transAxes)

    legend = axSPhP.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.05, 0.95), edgecolor='black', ncol=4)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    axFP.plot(wArrFP, dosFP, color = cmapBone(0.45), lw = 1.2)
    axFP.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    axFP.set_xlim(0., 1e14)
    axFP.set_ylim(0., 2.1)
    w0 = np.pi * consts.c / dFP[0]
    axFP.set_xticks([0., w0, 3. * w0, 5. * w0, 7. * w0, 9. * w0])
    axFP.set_xticklabels([r"$0$", r"$\omega_0$", r"$3\omega_0$", r"$5\omega_0$", r"$7\omega_0$", r"$9\omega_0$"])
    axFP.set_yticks([0., 2. / 3., 2])
    axFP.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axFP.set_xlabel(r"$\omega$")
    axFP.set_ylabel(r"$\rho_{||} / \rho_0$")

    axFP.text(-0.11, 1.1, r"$(\mathrm{a})$", fontsize=8, transform=axFP.transAxes)

    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf
    unitFac = consts.m_e / (consts.hbar * wLO / consts.c**2)

    axM.plot(zArrM / lambda0, np.abs(mArr) * unitFac, color = cmapPink(0.525), lw = 1.2, label = r"$\mathrm{Total}$")
    axM.plot(zArrM / lambda0, np.abs(mBulkArr) * unitFac, color = cmapPink(0.2), lw = 1.2, label = r"$\mathrm{Bulk}$")

    axM.set_xscale("log")
    axM.set_yscale("log")
    axM.set_xlim(1e-4, 1e1)
    axM.set_ylim(1e-4, 1e0)

    axM.set_xlabel(r"$z_0 [\lambda_0]$", fontsize = 8)
    axM.set_ylabel(r"$\Delta m \left[\frac{\hbar \omega_{\infty}}{c^2} \right]$", fontsize = 8, labelpad = 0)

    axM.plot(zArrM / lambda0, np.abs(mArr[-1]) * zArrM[-1] ** 3 / zArrM ** 3 * unitFac, color="red",
                lw=.4, linestyle = "-", label = r"$\sim z^{-3}$")

    axM.text(-0.15, 1.1, r"$(\mathrm{c})$", fontsize=8, transform=axM.transAxes)

    legend = axM.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.02, 1.02), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axSPhP.spines[axis].set_linewidth(.5)
#        axinSPhP.spines[axis].set_linewidth(.5)
        axFP.spines[axis].set_linewidth(.5)
        axM.spines[axis].set_linewidth(.5)

    plt.savefig("./PaperPlots/DosComparison.png")

def plotSquaredDispersion():

    xArr = np.linspace(-1., 1., 1000, endpoint = True)
    yArr = xArr ** 2

    fig = plt.figure(figsize=(2.2, 1.2), dpi=300)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=1., bottom=0., left=0.0, right=.95)

    ax = plt.subplot(gs[0, 0])

    ax.plot(xArr, 1.1 * yArr, color = 'red', lw = 1.5, linestyle = "--", label = r"$\varepsilon_{\rm eff}$", zorder = 666)
    ax.plot(xArr, yArr, color = 'black', lw = 1.5, label = r"$\varepsilon_0$")


    ax.set_xticks([])
    ax.set_yticks([])


    ax.arrow(0., -0.1, 0., 1.15, length_includes_head=True,
              head_width=0.04, head_length=0.04, linewidth = 1., color = "black")
    ax.arrow(-1., 0., 2., 0., length_includes_head=True,
              head_width=0.04, head_length=0.04, linewidth = 1., color = "black")

    ax.text(1.03, -0.03, r"$k$", fontsize = 20)
    #ax.text(0.05, 0.92, r"$\varepsilon(k)$", fontsize = 20)

    legend = ax.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.95, 1.2), edgecolor='black', ncol=1,
                       handlelength=0.8, handletextpad=0.2, labelspacing = 0.)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.0)

    plt.savefig("./PaperPlots/QuadDispersionIncrease.png")

def plotSquaredDispersion2():

    xArr = np.linspace(-1., 1., 1000, endpoint = True)
    yArr = xArr ** 2

    fig = plt.figure(figsize=(2.2, 1.2), dpi=300)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=1., bottom=0., left=0.0, right=.95)

    ax = plt.subplot(gs[0, 0])

    ax.plot(xArr, 0.7 * yArr, color = 'red', lw = 1.5, linestyle = "--", label = r"$\varepsilon_{\rm eff}$", zorder = 666)
    ax.plot(xArr, yArr, color = 'black', lw = 1.5, label = r"$\varepsilon_0$")


    ax.set_xticks([])
    ax.set_yticks([])


    ax.arrow(0., -0.1, 0., 1.1, length_includes_head=True,
              head_width=0.04, head_length=0.04, linewidth = 1., color = "black")
    ax.arrow(-1., 0., 2., 0., length_includes_head=True,
              head_width=0.04, head_length=0.04, linewidth = 1., color = "black")

    ax.text(1.03, -0.03, r"$k$", fontsize = 20)
    #ax.text(0.05, 0.85, r"$\varepsilon(k)$", fontsize = 20)

    legend = ax.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.95, 1.2), edgecolor='black', ncol=1,
                       handlelength=0.8, handletextpad=0.2, labelspacing = 0.)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.0)

    plt.savefig("./PaperPlots/QuadDispersionDecrease.png")


def main():

    #plotSquaredDispersion()
    #plotSquaredDispersion2()

    #plotEffectiveMassesComparison()
    #plotEffectiveMassesComparison_V2()
    createDosPlot()
    #plotClassicalAnalogy()

    #EffectiveMassFP()
    #EffectiveMassSPhP()


main()