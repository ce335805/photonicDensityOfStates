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

def plotEffectiveMassesComparison():

    wLO = 32.04 * 1e12
    wTO = 1e6
    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPhP = "./savedData/PaperData/massThesisFinal" + wLOStr + wTOStr + ".hdf5"
    cutoff, zArrSPP, massSPPArr = prodV2.readMasses(filenameSPhP)
    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf
    unitFac = consts.m_e / (consts.hbar * wLO / consts.c**2)

    fig = plt.figure(figsize=(3.3, 3.), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                           wspace=0.4, hspace=0.4, top=0.9, bottom=0.15, left=0.25, right=0.9)

    axIm = plt.subplot(gs[0, 0])
    axZoom = plt.subplot(gs[1, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axZoom.plot(zArrSPP / lambda0, np.abs(massSPPArr) * unitFac, color = cmapBone(.25), lw = 0.8, label = r"$\omega_{\rm p} = 32 \mathrm{THz}$")

    axZoom.plot(zArrSPP / lambda0, np.abs(massSPPArr[-1]) * zArrSPP[-1]**3 / zArrSPP**3 * unitFac, color = "red", lw = .4)
    fitInd = 150
    axZoom.plot(zArrSPP / lambda0, np.abs(massSPPArr[fitInd]) * zArrSPP[fitInd]**1 / zArrSPP**1 * unitFac, color = "red", lw = .4)

    axZoom.set_xscale("log")
    axZoom.set_yscale("log")

    axZoom.set_xlim(1e-2, 1e1)
    axZoom.set_ylim(1e-2, 1e4)

    axZoom.set_xlabel("$z [\lambda_0]$", fontsize = 8)
    axZoom.set_ylabel(r"$|\Delta m| \, \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \omega_{\rm p}}{c^2} \right]$", labelpad = 6., fontsize = 8)

    axZoom.text(0.25, 0.8, r"$\sim z^{-3}$", transform = axZoom.transAxes)
    axZoom.text(0.72, 0.25, r"$\sim z^{-1}$", transform = axZoom.transAxes)

    #legend = axZoom.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    axIm.set_xticks([])
    axIm.set_yticks([])
    arr_img = mpimg.imread('./PaperPlots/pngMassComic.png')
    im = OffsetImage(arr_img, zoom = .3)
    ab = AnnotationBbox(im, (0.1, -0.2), xycoords='axes fraction', box_alignment=(0, 0), frameon=False, pad=0, boxcoords="offset points")
    axIm.add_artist(ab)

    axIm.text(-0., 1.04, r"$\mathrm{(a)}$", fontsize = 8, transform = axIm.transAxes)
    axZoom.text(-0., 1.04, r"$\mathrm{(b)}$", fontsize = 8, transform = axZoom.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axZoom.spines[axis].set_linewidth(.5)
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

    wArrSPhP, dosSPhP = getDosDataSPP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

    dFP, wArrFP, dosFP = getDosFP()

    plotDos(zArr, wArrSPhP, dosSPhP, dFP, wArrFP, dosFP, wLO, wTO)

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

    return (wArrOne, dosTETotal + dosTMPara + dosSurf)


def plotDos(zArr, wArrSPhP, dosSPhP, dFP, wArrFP, dosFP, wLO, wTO):
    wInf = np.sqrt(wLO ** 2 + wTO ** 2) / np.sqrt(1 + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    fig = plt.figure(figsize=(7., 1.8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2],
                           wspace=0.2, hspace=0., top=0.85, bottom=0.25, left=0.08, right=0.95)

    axSPhP = plt.subplot(gs[0, 0])
    axFP = plt.subplot(gs[0, 1])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    indArr = np.array([1, 30, 60, 80, 100], dtype=int)

    # axSPhP.plot(wArr, dos[:, indArr[0]], color=cmapBone(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP, dosSPhP[:, indArr[1]], color=cmapBone(0.1), lw=.7,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP, dosSPhP[:, indArr[2]], color=cmapBone(0.3), lw=.7,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP, dosSPhP[:, indArr[3]], color=cmapBone(0.5), lw=.7,
            label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP, dosSPhP[:, indArr[4]], color=cmapBone(0.7), lw=.7,
            label="$z = $" + "{:1.1f}".format(zArr[indArr[4]] / lambda0) + r"$\lambda_0$")
    #
    axSPhP.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    axSPhP.set_xlim(0, 60 * 1e12)
    axSPhP.set_ylim(0., 3.)

    axSPhP.set_xticks([0., wTO, wInf, wLO])
    axSPhP.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize=8)
    axSPhP.set_yticks([0., 2. / 3., 2])
    axSPhP.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axSPhP.set_xlabel(r"$\omega$")
    axSPhP.set_ylabel(r"$\rho / \rho_0$")

    axinSPhP = axSPhP.inset_axes([0.55, 0.44, 0.4, 0.5])

    axinSPhP.plot(wArrSPhP, dosSPhP[:, indArr[1]], color=cmapBone(0.1), lw=.7,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    axinSPhP.plot(wArrSPhP, dosSPhP[:, indArr[2]], color=cmapBone(0.3), lw=.7,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    axinSPhP.plot(wArrSPhP, dosSPhP[:, indArr[3]], color=cmapBone(0.5), lw=.7,
            label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    axinSPhP.plot(wArrSPhP, dosSPhP[:, indArr[4]], color=cmapBone(0.7), lw=.7,
            label="$z = $" + "{:1.2f}".format(zArr[indArr[4]] / lambda0) + r"$\lambda_0$")

    axinSPhP.set_xlim(0, 35 * 1e12)
    axinSPhP.set_ylim(0., 100.)

    axinSPhP.set_xticks([0., wTO, wInf, wLO])
    axinSPhP.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize=6)
    axinSPhP.set_yticks([0., 100])
    axinSPhP.set_yticklabels([r"$0$", r"$100$"], fontsize=6)

    axinSPhP.set_xlabel(r"$\omega$", fontsize = 6, labelpad = -2)
    axinSPhP.set_ylabel(r"$\rho / \rho_0$", fontsize = 0)

    axSPhP.text(-0.11, 1.07, r"$(\mathrm{a})$", fontsize=8, transform=axSPhP.transAxes)

    legend = axSPhP.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.05, 0.95), edgecolor='black', ncol=4)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    axFP.plot(wArrFP, dosFP, color = cmapPink(0.45), lw = 0.7)
    axFP.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    axFP.set_xlim(0., 1e14)
    axFP.set_ylim(0., 2.1)
    w0 = np.pi * consts.c / dFP[0]
    axFP.set_xticks([0., w0, 3. * w0, 5. * w0, 7. * w0, 9. * w0])
    axFP.set_xticklabels([r"$0$", r"$\omega_0$", r"$3\omega_0$", r"$5\omega_0$", r"$7\omega_0$", r"$9\omega_0$"])
    axFP.set_yticks([0., 2. / 3., 2])
    axFP.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axFP.set_xlabel(r"$\omega$")
    axFP.set_ylabel(r"$\rho / \rho_0$")

    axFP.text(-0.11, 1.07, r"$(\mathrm{b})$", fontsize=8, transform=axFP.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axSPhP.spines[axis].set_linewidth(.5)
        axinSPhP.spines[axis].set_linewidth(.5)
        axFP.spines[axis].set_linewidth(.5)

    plt.savefig("./PaperPlots/DosComparison.png")


def main():
    #plotEffectiveMassesComparison()
    createDosPlot()
main()