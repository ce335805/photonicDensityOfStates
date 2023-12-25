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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

import epsilonFunctions as epsFunc
import produceFreqData as prod
import produceFreqDataV2 as prodV2
import dosFuncs.dosTMSurfModes as surfModes
import dosAsOfFreq

def plotDosWhole(zArr, wLO, wTO, epsInf, L):

    wArr = prodV2.defineFreqArray()

    dosTETotal = prodV2.retrieveDosTE(wArr, L, wLO, wTO, epsInf)
    dosTMPara = prodV2.retrieveDosTMPara(wArr, L, wLO, wTO, epsInf)

    dosSurf = np.zeros((len(wArr), len(zArr)))
    for wInd, wVal in enumerate(wArr):
        epsilon = epsFunc.epsilon(wVal, wLO, wTO, epsInf)
        dosSurf[wInd, :] = 1. / (1. + np.abs(epsilon)) * dosAsOfFreq.getDosTMSurf(wVal, zArr, L, wLO, wTO, epsInf)

    filename = "Para"
    createDosPlotFreq(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosPlotNatUnits(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosPlotFreqThesis(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosPlotFreqThesiswTO(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosRealSpaceThesis(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf, L)

def plotDosWholeOld(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)
    wArr = prodV2.defineFreqArray()

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, wLO, wTO, epsInf)
    dosTMPara = prod.retrieveDosTMPara(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, wLO, wTO, epsInf)

    dosSurf = np.zeros((len(wArr), len(zArr)))
    for wInd, wVal in enumerate(wArr):
        epsilon = epsFunc.epsilon(wVal, wLO, wTO, epsInf)
        dosSurf[wInd, :] = 1. / (1. + np.abs(epsilon)) * dosAsOfFreq.getDosTMSurf(wVal, zArr, L, wLO, wTO, epsInf)

    filename = "Para"
    #createDosPlotFreq(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosPlotNatUnits(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosPlotFreqThesis(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    createDosPlotFreqThesiswTO(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    #createDosRealSpaceThesis(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf, L)


def createDosPlotFreq(wArr, zArr, dos, filename, wLO, wTO, epsInf):
    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.2, left=0.13, right=0.96)
    ax = plt.subplot(gs[0, 0])

    #arr_img = mpimg.imread('SPhPV3.png')
    #im = OffsetImage(arr_img, zoom = .1)
    #ab = AnnotationBbox(im, (1.0, 0.15), xycoords='axes fraction', box_alignment=(1.05, -0.05), frameon=False, pad=0, boxcoords="offset points")
    #ax.add_artist(ab)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    cutoff = 3000 * 1e12
    cutoffFac = np.exp(- wArr ** 2 / cutoff ** 2)

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)

    lambda0 = 2. * np.pi * consts.c / wInf

    indArr = np.array([1, 40, 80, 120, 181], dtype = int)
    #ax.plot(wArr, (dos[:, indArr[0]] - 4. / 6.) * wArr**3 * cutoffFac, color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[0]] * wInf / consts.c))
    #ax.plot(wArr, (dos[:, indArr[1]] - 4. / 6.) * wArr**3 * cutoffFac, color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[1]] * wInf / consts.c))
    #ax.plot(wArr, (dos[:, indArr[2]] - 4. / 6.) * wArr**3 * cutoffFac, color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[2]] * wInf / consts.c))
    #ax.plot(wArr, (dos[:, indArr[3]] - 4. / 6.) * wArr**3 * cutoffFac, color=cmapPink(0.6), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[3]] * wInf / consts.c))
    #ax.plot(wArr, (dos[:, indArr[4]] - 4. / 6.) * wArr**3 * cutoffFac, color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[4]] * wInf / consts.c))


    ax.plot(wArr, dos[:, indArr[0]], color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[1]], color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[2]], color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[3]], color=cmapPink(0.6), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[4]], color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[4]] * wInf / consts.c) + r"$\lambda_0$")
#
    #ax.axhline(0, lw = 0.5, color = 'gray', zorder = -666)

    #ax.set_xlim(500 * 1e12, 1000 * 1e12)
    ax.set_ylim(0, 5)

    #ax.set_xticks([0., wLO, 2. * wLO])
    #ax.set_xticklabels([r"$0$", r"$\omega_{\rm LO}$", r"$2 \omega_{\rm LO}$"])


    ax.set_xlabel(r"$\omega \, [\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0., 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)



    plt.savefig("./SPhPPlotsSaved/dosAsOfFreq" + filename + ".png")

def createDosPlotNatUnits(wArr, zArr, dos, filename, wLO, wTO, epsInf):
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    natUnitsFac = wLO * consts.c ** 3 * (2. * np.pi)**3 / (wInf ** 3)

    lambda0 = 2. * np.pi * consts.c / wInf

    indArr = np.array([1, 11, 20, 30, 40], dtype = int)
    ax.plot(wArr, (dos[:, indArr[0]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, (dos[:, indArr[1]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, (dos[:, indArr[2]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, (dos[:, indArr[3]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.6), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    #ax.plot(wArr, (dos[:, indArr[4]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[4]] * wInf / consts.c))

    #ax.set_xlim(np.amin(wArr), 5. * wLO * 1e-12)
    ax.set_xlim(np.amin(wArr), np.amax(wArr))
    ax.set_xlim(0, 5. * wLO)
    ax.set_ylim(-0.03 * (2. * np.pi)**3, 0.2 * (2. * np.pi)**3)

    ax.set_xticks([0., wLO, 2. * wLO])
    ax.set_xticklabels([r"$0$", r"$\omega_{\rm LO}$", r"$2 \omega_{\rm LO}$"])


    ax.set_xlabel(r"$\omega \, [\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho - \rho_0 \, \left[\frac{1}{\omega_{\rm LO} \lambda_0^3}\right]$")

    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosAsOfFreq" + filename + "NatUnits.png")


def createDosPlotFreqThesis(wArr, zArr, dos, filename, wLO, wTO, epsInf):
    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.12, right=0.95)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    indArr = np.array([1, 30, 60, 80, 90], dtype = int)

    #ax.plot(wArr, dos[:, indArr[0]], color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[1]], color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[2]], color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[3]], color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[4]], color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[4]] / lambda0) + r"$\lambda_0$")
#
    #ax.axhline(0, lw = 0.5, color = 'gray', zorder = -666)

    ax.set_xlim(0, 1.4 * wLO)
    ax.set_ylim(0, 5)

    ax.set_xticks([0., wInf, wLO])
    ax.set_xticklabels([r"$0$", r"$\omega_{\infty}$", r"$\omega_{\rm p}$"], fontsize = 8)
    ax.set_yticks([0., 2./3., 2, 4])
    ax.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$", r"$4$"], fontsize = 8)


    ax.set_xlabel(r"$\omega $")
    ax.set_ylabel(r"$\rho / \rho_0$")

    ax.text(-0.11, 1.07, r"$(\mathrm{a})$", fontsize = 8, transform = ax.transAxes)

    legend = ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/dosAsOfFreq" + filename + ".png")

def createDosPlotFreqThesiswTO(wArr, zArr, dos, filename, wLO, wTO, epsInf):
    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.12, right=0.95)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    indArr = np.array([1, 30, 60, 80, 90], dtype = int)

    #ax.plot(wArr, dos[:, indArr[0]], color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[1]], color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[2]], color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[3]], color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, dos[:, indArr[4]], color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[4]] / lambda0) + r"$\lambda_0$")
#
    #ax.axhline(0, lw = 0.5, color = 'gray', zorder = -666)

    ax.set_xlim(0, 1.4 * wLO)
    ax.set_ylim(0, 5)

    ax.set_xticks([0., wTO, wInf, wLO])
    ax.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)
    ax.set_yticks([0., 2./3., 2, 4])
    ax.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$", r"$4$"], fontsize = 8)


    ax.set_xlabel(r"$\omega $")
    ax.set_ylabel(r"$\rho / \rho_0$")

    ax.text(-0.11, 1.07, r"$(\mathrm{b})$", fontsize = 8, transform = ax.transAxes)

    legend = ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/dosAsOfFreqwTO" + filename + ".png")

def createDosPlotNatUnitsThesis(wArr, zArr, dos, filename, wLO, wTO, epsInf):
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    natUnitsFac = wLO * consts.c ** 3 * (2. * np.pi)**3 / (wInf ** 3)

    lambda0 = 2. * np.pi * consts.c / wInf

    indArr = np.array([1, 11, 20, 30, 40], dtype = int)
    ax.plot(wArr, (dos[:, indArr[0]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, (dos[:, indArr[1]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, (dos[:, indArr[2]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    ax.plot(wArr, (dos[:, indArr[3]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.6), lw=.7, label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    #ax.plot(wArr, (dos[:, indArr[4]] - 4. / 6.) * wArr**2 / (np.pi**2 * consts.c**3) * natUnitsFac, color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[4]] * wInf / consts.c))

    #ax.set_xlim(np.amin(wArr), 5. * wLO * 1e-12)
    ax.set_xlim(np.amin(wArr), np.amax(wArr))
    ax.set_xlim(0, 5. * wLO)
    ax.set_ylim(-0.03 * (2. * np.pi)**3, 0.2 * (2. * np.pi)**3)

    ax.set_xticks([0., wLO, 2. * wLO])
    ax.set_xticklabels([r"$0$", r"$\omega_{\rm LO}$", r"$2 \omega_{\rm LO}$"])


    ax.set_xlabel(r"$\omega \, [\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho - \rho_0 \, \left[\frac{1}{\omega_{\rm LO} \lambda_0^3}\right]$")

    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosAsOfFreq" + filename + "NatUnits.png")

def createDosRealSpaceThesis(wArr, zArr, dos, filename, wLO, wTO, epsInf, L):
    fig = plt.figure(figsize=(4.5, 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.15, right=0.92)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    indArr = np.array([2, 6, 7, 9], dtype = int)
    epsArr = np.zeros(len(indArr))

    for loopInd, wInd in enumerate(indArr):
        color = cmapPink((loopInd + 1) / (len(indArr) + 2.5))
        ax.plot(zArr, dos[wInd, :], color=color, lw=.7, label="$\omega = $" + "{:1.1f}".format(wArr[wInd] / wLO) + r"$\omega_{\rm LO}$")

        epsArr[loopInd] = epsFunc.epsilon(wArr[wInd], wLO, wTO, epsInf)
        #if (epsArr[loopInd] > 0):
        #    ax.axhline(np.sqrt(epsArr[loopInd]) * 2. / 3., color = color, lw = .5)

    ax.axvline(0, color = "black", lw = .4, zorder = -1)

    ax.set_xlim(- L / 38., L / 28)
    ax.set_ylim(0, 5)

    ax.set_xticks([- 20. * consts.c / wLO, - 10. * consts.c / wLO, 0., 10. * consts.c / wLO, 20. * consts.c / wLO])
    ax.set_xticklabels([r"$-\frac{20 \, c}{\omega_{\rm LO}}$", r"$-\frac{10 \, c}{\omega_{\rm LO}}$", r"$0$", r"$\frac{10 \, c}{\omega_{\rm LO}}$", r"$\frac{20 \, c}{\omega_{\rm LO}}$"], fontsize = 8)
    ax.set_yticks([0.])
    ax.set_yticklabels([r"$0$"], fontsize = 8)

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\rho / \rho_0$", rotation = 0)
    ax.yaxis.set_label_coords(0, 1)

    #ax.annotate('', xy=(- L / 12, np.sqrt(epsArr[0]) * 2. / 3.), xytext=(- L / 20, np.sqrt(epsArr[0]) * 2. / 3.),
    #            arrowprops=dict(facecolor='red', arrowstyle='<-'))

    #ax.arrow(-L / 10, np.sqrt(epsArr[0]) * 2. / 3., 0, 0, clip_on = False, width = .001, head_width = 0.1, head_length =.001, color = cmapPink(1 /(len(indArr) + 1)))
    ax.arrow(- 0.00030, np.sqrt(epsArr[0]) * 2. / 3., 0.00002, 0, head_length = 0.00001, head_width = 0.15, clip_on = False, color = cmapPink(1 /(len(indArr) + 2.5)))
    ax.arrow(- 0.00030, np.sqrt(epsArr[-1]) * 2. / 3., 0.00002, 0, head_length = 0.00001, head_width = 0.15, clip_on = False, color = cmapPink(4 /(len(indArr) + 2.5)))
    ax.arrow( 0.000395, 2. / 3., -0.00002, 0, head_length = 0.00001, head_width = 0.15, clip_on = False, color = "gray")

    ax.text(- 0.00037, np.sqrt(epsArr[0]) * 2. / 3. - 0.2, r"$\frac{2}{3}\sqrt{\varepsilon(\omega_1)}$", fontsize = 6)
    ax.text(- 0.00037, np.sqrt(epsArr[-1]) * 2. / 3. - 0.2, r"$\frac{2}{3} \sqrt{\varepsilon(\omega_4)}$", fontsize = 6)
    ax.text(0.0004, 2. / 3. - 0.25, r"$\frac{2}{3}$", fontsize = 6)

    ax.text(- 0.0002, 4.5, r"$\varepsilon(\omega)$", fontsize = 8)
    ax.text(0.00005, 4.5, r"$\varepsilon = 1$", fontsize = 8)


    #ax.text(- 0.00095, 5.2, r"$(\mathrm{b})$", fontsize = 8)
    ax.text(-0.18, 1.05, r"$(\mathrm{b})$", fontsize = 8, transform = ax.transAxes)

    #legend = ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    ### add an inset ###

    omegaArr = np.linspace(0, 40 * 1e12, 1000)
    eps = (wLO ** 2 - omegaArr ** 2) / (wTO ** 2 - omegaArr ** 2)

    ax_inset = inset_axes(ax, width="100%", height="100%", loc='upper right',bbox_to_anchor=(.6,.38,.4,.45), bbox_transform=ax.transAxes)

    ax_inset.plot(omegaArr, eps, color = cmapBone(.3), lw = .4)
    ymin = -20
    ymax = 40
    ax_inset.set_ylim(ymin, ymax)
    ax_inset.set_xlim(np.amin(omegaArr), np.amax(omegaArr))

    ax_inset.axhline(0, color = "black", lw = .3)

    ax_inset.axvline(wArr[indArr[0]], lw = .5, color = cmapPink(1 /(len(indArr) + 2.5)))
    ax_inset.axvline(wArr[indArr[1]], lw = .5, color = cmapPink(2 /(len(indArr) + 2.5)))
    ax_inset.axvline(wArr[indArr[2]], lw = .5, color = cmapPink(3 /(len(indArr) + 2.5)))
    ax_inset.axvline(wArr[indArr[3]], lw = .5, color = cmapPink(4 /(len(indArr) + 2.5)))

    ax_inset.text(wArr[indArr[0]], ymax + 2, r"$\omega_1$", fontsize = 6, horizontalalignment='center')
    ax_inset.text(wArr[indArr[1]], ymax + 2, r"$\omega_2$", fontsize = 6, horizontalalignment='center')
    ax_inset.text(wArr[indArr[2]], ymax + 2, r"$\omega_3$", fontsize = 6, horizontalalignment='center')
    ax_inset.text(wArr[indArr[3]], ymax + 2, r"$\omega_4$", fontsize = 6, horizontalalignment='center')

    ax_inset.set_ylabel(r"$\varepsilon(\omega)$", labelpad=0, fontsize = 6)
    ax_inset.yaxis.set_label_coords(-0.07, .6)

    ax_inset.set_yticks([0])
    ax_inset.set_yticklabels([r"$0$"], fontsize = 6)
    ax_inset.set_xticks([0, wTO, wInf, wLO])
    ax_inset.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 6)
    ax_inset.tick_params(axis='both', which='major', pad=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        ax_inset.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/dosRealSpace" + filename + ".png")




