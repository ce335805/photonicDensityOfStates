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

import findAllowedKsSPhP as findAllowedKsSPhP
import epsilonFunctions as epsFunc

import wfFuncs.TEWavefunctionSPhP as TEWF
import wfFuncs.TEWavefunctionEvaSPhP as TEEvaWF
import wfFuncs.TEWavefunctionResSPhP as TEResWF
import wfFuncs.TMWavefunctionSPhP as TMWF
import wfFuncs.TMWavefunctionEvaSPhP as TMEvaWF
import wfFuncs.TMWavefunctionResSPhP as TMResWF
import wfFuncs.TMWavefunctionSurf as TMSurf

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



def plotWFs(zArr,
            wfTE,
            wfTEEva,
            wfTERes,
            wfTMPara,
            wfTMPerp,
            wfTMEvaPara,
            wfTMEvaPerp,
            wfTMResPara,
            wfTMResPerp,
            wLO,
            wTO,
            L):

    fig = plt.figure(figsize=(6.5, 2.), dpi=800)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                           wspace=0.2, hspace=0.0, top=0.9, bottom=0.2, left=0.05, right=0.99)
    axTE = plt.subplot(gs[0, 0])
    axTEEva = plt.subplot(gs[0, 1])
    axTERes = plt.subplot(gs[0, 2])
    axTM = plt.subplot(gs[1, 0])
    axTMEva = plt.subplot(gs[1, 1])
    axTMRes = plt.subplot(gs[1, 2])

    axList = np.array([axTE, axTEEva, axTERes, axTM, axTMEva, axTMRes])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    lwVal = 1.2

    axTE.plot(zArr, wfTE[2, ], color = cmapPink(.45), lw = lwVal)
    axTEEva.plot(zArr, wfTEEva[1, ], color = cmapPink(.45), lw = lwVal)
    axTERes.plot(zArr, wfTERes[2, ], color = cmapPink(.45), lw = lwVal)
    axTM.plot(zArr, wfTMPara[2, ], color = cmapBone(.3), lw = lwVal, label = r"$f_{||}$")
    axTM.plot(zArr, wfTMPerp[2, ], color = cmapBone(.7), lw = lwVal, label = r"$f_{\perp}$")
    axTMEva.plot(zArr, wfTMEvaPara[3, ], color = cmapBone(.3), lw = lwVal, label = r"$f_{||}$")
    axTMEva.plot(zArr, wfTMEvaPerp[3, ], color = cmapBone(.7), lw = lwVal, label = r"$f_{\perp}$")
    axTMRes.plot(zArr, wfTMResPara[3, ], color = cmapBone(.3), lw = lwVal, label = r"$f_{||}$")
    axTMRes.plot(zArr, wfTMResPerp[3, ], color = cmapBone(.7), lw = lwVal, label = r"$f_{\perp}$")

    for ax in axList:
        ax.axvline(0, color="gray", lw=0.4)
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlim(-L / 2, L / 2)
        ax.set_yticks([0])
        ax.set_yticks([0])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.fill_between(zArr, y1=ymin, y2=ymax, where=(zArr <= 0), color='oldlace', alpha=1., zorder=-667)

    axTE.set_ylabel("$f(z)$", fontsize = 8)
    axTM.set_ylabel("$f(z)$", fontsize = 8)

    for ax in axList[:3]:
        ax.set_xticks([])
    for ax in axList[3:]:
        ax.set_xticks([- L / 2, 0, L / 2])
        ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"], fontsize = 8)
        ax.set_xlabel("$z$", fontsize = 8)

    axTE.text(0.12, 1.05,r"$\varepsilon(\omega) > 1$",transform=axTE.transAxes, fontsize = 8)
    axTEEva.text(0.12, 1.05,r"$\varepsilon(\omega) > 1$",transform=axTEEva.transAxes, fontsize = 8)
    axTERes.text(0.12, 1.05,r"$\varepsilon(\omega) < 1$",transform=axTERes.transAxes, fontsize = 8)

    legend = axTMRes.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0., 1.05), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.05)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        for ax in axList:
            ax.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/combinedWFPlot.png")



def plotSPhP(zArr,
            wfPara,
            wfPerp,
            wLO,
            wTO,
            L):

    fig = plt.figure(figsize=(2., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.2, hspace=0.0, top=0.9, bottom=0.25, left=0.2, right=0.95)

    ax = plt.subplot(gs[0, 0])


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    lwVal = 1.2

    ax.plot(zArr, wfPara, color = cmapBone(.3), lw = lwVal, label = r"$f_{||}$")
    ax.plot(zArr, wfPerp, color = cmapBone(.7), lw = lwVal, label = r"$f_{\perp}$")

    ax.axvline(0, color="gray", lw=0.4)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlim(-L / 4500, L / 4500)
    ax.set_yticks([0])
    ax.set_yticks([0])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.fill_between(zArr, y1=ymin, y2=ymax, where=(zArr <= 0), color='oldlace', alpha=1., zorder=-667)

    ax.set_ylabel("$f(z)$", fontsize = 8)

    ax.set_xticks([- L / 5000, 0, L / 5000])
    ax.set_xticklabels([r"$-\frac{L}{5000}$", r"$0$", r"$\frac{L}{5000}$"], fontsize = 8)
    ax.set_xlabel("$z$", fontsize = 8)

    ax.text(0.58, .85,r"$\omega = 0.9\omega_{\infty}$",transform=ax.transAxes, fontsize = 8)
    ax.text(0.12, 1.05,r"$\varepsilon(\omega) < 0$",transform=ax.transAxes, fontsize = 8)

    legend = ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0., 1.05), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.05)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/wfSPhP.png")


def TEWavefunctions(zArr, L, wLO, wTO, epsInf):
    omega = 1. * 1e10
    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TE")
    kArr = allowedKs[0 : 5]
    wF = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    for kInd, kVal in enumerate(kArr):
        wF[kInd, :] = TEWF.waveFunctionTE(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return wF

def TEEvaWavefunctions(zArr, L, wLO, wTO, epsInf):
    omega = 1. * 1e10
    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TEEva")
    kArr = allowedKs[0 : 5]
    wF = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    for kInd, kVal in enumerate(kArr):
        wF[kInd, :] = TEEvaWF.waveFunctionTEEva(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return wF


def TEResWavefunctions(zArr, L, wLO, wTO, epsInf):
    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2.)
    omega = 0.95 * wInf
    #omega = 10. * wLO
    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TERes")
    kArr = allowedKs[0 : 5]
    wF = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    for kInd, kVal in enumerate(kArr):
        wF[kInd, :] = TEResWF.waveFunctionTERes(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return wF


def TMWavefunctions(zArr, L, wLO, wTO, epsInf):
    omega = 1. * 1e10
    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")
    kArr = allowedKs[0 : 5]
    wFPara = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    wFPerp = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    for kInd, kVal in enumerate(kArr):
        wFPara[kInd, :] = TMWF.waveFunctionTMPara(zArr, kVal, L, omega, wLO, wTO, epsInf)
        wFPerp[kInd, :] = TMWF.waveFunctionTMPerp(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return (wFPara, wFPerp)

def TMEvaWavefunctions(zArr, L, wLO, wTO, epsInf):
    omega = 1. * 1e10
    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TMEva")
    kArr = allowedKs[0 : 5]
    wFPara = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    wFPerp = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    for kInd, kVal in enumerate(kArr):
        wFPara[kInd, :] = TMEvaWF.waveFunctionTMPara(zArr, kVal, L, omega, wLO, wTO, epsInf)
        wFPerp[kInd, :] = TMEvaWF.waveFunctionTMPerp(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return (wFPara, wFPerp)

def TMResWavefunctions(zArr, L, wLO, wTO, epsInf):
    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2.)
    omega = .99 * wInf
    allowedKs = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TMRes")
    kArr = allowedKs[0 : 5]
    wFPara = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    wFPerp = np.zeros((kArr.shape[0], zArr.shape[0]), dtype=float)
    for kInd, kVal in enumerate(kArr):
        wFPara[kInd, :] = TMResWF.waveFunctionTMPara(zArr, kVal, L, omega, wLO, wTO, epsInf)
        wFPerp[kInd, :] = TMResWF.waveFunctionTMPerp(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return (wFPara, wFPerp)

def TMSurfWavefunctions(zArr, L, wLO, wTO, epsInf):
    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2.)
    omega = .9 * wInf
    wFPara = np.zeros((zArr.shape[0]), dtype=float)
    wFPerp = np.zeros((zArr.shape[0]), dtype=float)
    kVal = omega / (consts.c * np.sqrt(np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf)) - 1))
    wFPara[:] = TMSurf.waveFunctionTMPara(zArr, kVal, L, omega, wLO, wTO, epsInf)
    wFPerp[:] = TMSurf.waveFunctionTMPerp(zArr, kVal, L, omega, wLO, wTO, epsInf)

    return (wFPara, wFPerp)

def combinedWFMain():
    print("Plotting a big panel with wave-functions")

    wLO = 32.04 * 1e12
    wTO = 22. * 1e12
    L = 1.
    epsInf = 1.

    zArr = np.linspace(- L / 2., L / 2., 2000)

    #wfTE = TEWavefunctions(zArr, L, wLO, wTO, epsInf)
    #wfTEEva = TEEvaWavefunctions(zArr, L, wLO, wTO, epsInf)
    #wfTERes = TEResWavefunctions(zArr, L, wLO, wTO, epsInf)
    #wfTMPara, wfTMPerp = TMWavefunctions(zArr, L, wLO, wTO, epsInf)
    #wfTMEvaPara, wfTMEvaPerp = TMEvaWavefunctions(zArr, L, wLO, wTO, epsInf)
    #wfTMResPara, wfTMResPerp = TMResWavefunctions(zArr, L, wLO, wTO, epsInf)
    #plotWFs(zArr,
    #        wfTE,
    #        wfTEEva,
    #        wfTERes,
    #        wfTMPara,
    #        wfTMPerp,
    #        wfTMEvaPara,
    #        wfTMEvaPerp,
    #        wfTMResPara,
    #        wfTMResPerp,
    #        wLO,
    #        wTO,
    #        L)

    zArr = np.linspace(- L / 4500., L / 4500., 2000)
    wfSurfPara, wfSurfPerp = TMSurfWavefunctions(zArr, L, wLO, wTO, epsInf)
    plotSPhP(zArr, wfSurfPara, wfSurfPerp, wLO, wTO, L)

combinedWFMain()