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
import dosFuncs.dosTMSurfModes as surfModes
import dosAsOfFreq
import performFreqIntegral

def plotMassScaling(L):
    epsInf = 1.
    wTO = 1e6# metallic case
    wLOArr = np.array([10., 20., 30.]) * 1e12
    cutoff = 400. * 1e12

    ### create arrays - set some stuff just for sizes
    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLOArr[0], wTO, epsInf)
    wArrTemp = np.append(arrBelow, arrWithin)
    wArrTemp = np.append(wArrTemp, arrAbove)
    zArrLen = 1001
    wArrs = np.zeros((len(wLOArr), len(wArrTemp)))
    dosTETotal = np.zeros((len(wLOArr), len(wArrTemp), zArrLen))
    dosTMPara = np.zeros((len(wLOArr), len(wArrTemp), zArrLen))

    ### read in data
    for wLOInd, wLO in enumerate(wLOArr):
        arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
        wArrTemp = np.append(arrBelow, arrWithin)
        wArrTemp = np.append(wArrTemp, arrAbove)
        wArrs[wLOInd, :] = wArrTemp
        dosTETotal[wLOInd, :, :] = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, wLO, wTO, epsInf)
        dosTMPara[wLOInd, :, :] = prod.retrieveDosTMPara(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, wLO, wTO, epsInf)

    ### compute mass
    MassTETotal = np.zeros((len(wLOArr), zArrLen))
    MassTMPara = np.zeros((len(wLOArr), zArrLen))
    prefacMass = 16. / (3. * np.pi) * consts.hbar / (consts.c ** 2 * consts.m_e)
    for wLOInd, wLO in enumerate(wLOArr):
        wInf = wLO / np.sqrt(2)
        lambda0 = 2. * np.pi * consts.c / wInf
        zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), zArrLen - 1, endpoint=True, base=10)
        zArr = np.append([L / 4.], zArr)
        for zInd, zVal in enumerate(zArr):
            cutoffFac = np.exp(- wArrs[wLOInd, :] ** 2 / cutoff ** 2)
            intFuncTE = prefacMass * (dosTETotal[wLOInd, :, zInd] - .5) * cutoffFac
            MassTETotal[wLOInd, zInd] = np.trapz(intFuncTE, x=wArrs[wLOInd, :], axis=0)
            intFuncTM = prefacMass * (dosTMPara[wLOInd, :, zInd] - 1. / 6.) * cutoffFac
            MassTMPara[wLOInd, zInd] = np.trapz(intFuncTM, x=wArrs[wLOInd, :], axis=0)

    MassSurf = np.zeros((len(wLOArr), zArrLen))
    for wLOInd, wLO in enumerate(wLOArr):
        wInf = wLO / np.sqrt(2)
        lambda0 = 2. * np.pi * consts.c / wInf
        zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), zArrLen - 1, endpoint=True, base=10)
        zArr = np.append([L / 4.], zArr)
        for zInd, zVal in enumerate(zArr):
            MassSurf[wLOInd, zInd] = performFreqIntegral.performSPhPIntegralMass(zVal, wLO, wTO, epsInf)[0]

    masses = MassTETotal + MassTMPara + MassSurf
    filename = "Para"
    createMassScalingPlot(masses, wLOArr, L, zArrLen, filename)
    createMassScalingPlotLin(masses, wLOArr, L, zArrLen, filename)


def createMassScalingPlot(massArrs, wLOArr, L, zArrLen, filename):
    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.2, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')


    for wLOInd, wLO in enumerate(wLOArr):
        wInf = wLO / np.sqrt(2)
        lambda0 = 2. * np.pi * consts.c / wInf
        zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), zArrLen - 1, endpoint=True, base=10)
        zArr = np.append([L / 4.], zArr)
        color = cmapPink((wLOInd + 1.) / (len(wLOArr) + 2.))
        wFac = wLO / wLOArr[0]
        #wFac = 1.
        ax.plot(zArr[1:] / lambda0, np.abs(massArrs[wLOInd, 1:]) / wFac, color = color, lw = .8, label = r"$\omega_{\rm LO} = $" + "{}".format(wLO * 1e-12) + r"$\mathrm{THz}$")

        if(wLOInd == len(wLOArr) - 1):
            fitInd = np.argmin(np.abs(zArr - 10. * lambda0))
            ax.plot(zArr / lambda0, np.abs(massArrs[-1, fitInd] * zArr[fitInd] / zArr) / wFac, color = 'red', lw = .4)
            fitInd = np.argmin(np.abs(zArr - .1 * lambda0))
            ax.plot(zArr / lambda0, np.abs(massArrs[-1, fitInd] * zArr[fitInd]**3 / zArr**3) / wFac, color = 'red', lw = .4)


    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-11, 1e-5)

    ax.set_xlabel(r"$z \, [\lambda_0]$", fontsize = 8)
    ax.set_ylabel(r"$|\Delta m| \, \left[m_{\mathrm e} \times \frac{\omega_{\rm LO}}{10 \mathrm{THz}} \right]$")

    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)


    plt.savefig("./SPhPPlotsSaved/massesScaling" + filename + ".png")


def createMassScalingPlotLin(massArrs, wLOArr, L, zArrLen, filename):
    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')


    for wLOInd, wLO in enumerate(wLOArr):
        wInf = wLO / np.sqrt(2)
        lambda0 = 2. * np.pi * consts.c / wInf
        zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), zArrLen - 1, endpoint=True, base=10)
        zArr = np.append([L / 4.], zArr)
        color = cmapPink((wLOInd + 1.) / (len(wLOArr) + 2.))
        ax.plot(zArr[1:] / lambda0,massArrs[wLOInd, 1:], color = color, lw = .8, label = r"$\omega_{\rm LO} = $" + "{}".format(wLO * 1e-12) + r"$\mathrm{THz}$")

    ax.axhline(0., lw = .4, color = 'black')
    #ax.set_xscale("log")


    ax.set_xlim(0., 1e1)
    ax.set_ylim(- 0.5 * 1e-8, 0.5 * 1e-8)

    ax.set_xticks([0., 1., 2., 5., 10.])

    ax.set_xlabel(r"$z \, [\lambda_0]$", fontsize = 8)
    ax.set_ylabel(r"$\Delta m \, [m_{\mathrm e}]$")

    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)


    plt.savefig("./SPhPPlotsSaved/massesScalingLin" + filename + ".png")

