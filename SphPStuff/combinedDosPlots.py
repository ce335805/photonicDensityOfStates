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

def plotDosWhole(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMPara = prod.retrieveDosTMPara(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)

    #wMaxAboveArr = np.array([np.linspace(wLO, 50. * wLO, 5000, endpoint=False)[-1], arrAbove[-1]])
    #dosTETotal = prod.retrieveDosTEMultipleWMax(arrBelow[-1], arrWithin[-1], wMaxAboveArr, L, epsInf)
    #dosTMPara = prod.retrieveDosTMParaMultipleWMax(arrBelow[-1], arrWithin[-1], wMaxAboveArr, L, epsInf)

    dosSurf = np.zeros((len(wArr), len(zArr)))
    for wInd, wVal in enumerate(wArr):
        epsilon = epsFunc.epsilon(wVal, wLO, wTO, epsInf)
        dosSurf[wInd, :] = 1. / (1. + np.abs(epsilon)) * dosAsOfFreq.getDosTMSurf(wVal, zArr, L, wLO, wTO, epsInf)

    filename = "Para"
    #createDosPlotFreq(wArr, zArr, dosTMPara + dosTETotal + dosSurf, filename, wLO, wTO, epsInf)
    createDosPlotFreq(wArr, zArr, dosTETotal, filename, wLO, wTO, epsInf)

def createDosPlotFreq(wArr, zArr, dos, filename, wLO, wTO, epsInf):
    wArr = wArr * 1e-12
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    cutoff = 3000
    cutoffFac = np.exp(- wArr ** 2 / cutoff ** 2)

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)

    indArr = np.array([1, 8, 16, 24, 32], dtype = int)
    ax.plot(wArr, (dos[:, indArr[0]] - 3. / 6.) * wArr**2 * cutoffFac, color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[0]] * wInf / consts.c))
    ax.plot(wArr, (dos[:, indArr[1]] - 3. / 6.) * wArr**2 * cutoffFac, color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[1]] * wInf / consts.c))
    ax.plot(wArr, (dos[:, indArr[2]] - 3. / 6.) * wArr**2 * cutoffFac, color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[2]] * wInf / consts.c))
    ax.plot(wArr, (dos[:, indArr[3]] - 3. / 6.) * wArr**2 * cutoffFac, color=cmapPink(0.6), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[3]] * wInf / consts.c))
    ax.plot(wArr, (dos[:, indArr[4]] - 3. / 6.) * wArr**2 * cutoffFac, color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[4]] * wInf / consts.c))


#    ax.plot(wArr, dos[:, indArr[0]], color=cmapPink(0.1), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[0]] * wInf / consts.c))
#    ax.plot(wArr, dos[:, indArr[1]], color=cmapPink(0.3), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[1]] * wInf / consts.c))
    #ax.plot(wArr, dos[:, indArr[0]], color=cmapPink(0.1), lw=.7,
    #        label="$z = 80$" + r"$\lambda_0$")
    #ax.plot(wArr, dos[:, indArr[1]], color=cmapPink(0.3), lw=.7,
    #        label="$z = 10$" + r"$\lambda_0$")
    #ax.plot(wArr, dos[:, indArr[2]], color=cmapPink(0.5), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[2]] * wInf / consts.c) + r"$\lambda_0$")
    #ax.plot(wArr, dos[:, indArr[3]], color=cmapPink(0.6), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[3]] * wInf / consts.c) + r"$\lambda_0$")
    #ax.plot(wArr, dos[:, indArr[4]], color=cmapPink(0.7), lw=.7, label="$z = $" + "{:1.1g}".format(zArr[indArr[4]] * wInf / consts.c) + r"$\lambda_0$")
#
    #ax.axhline(0, lw = 0.5, color = 'gray', zorder = -666)
    ax.axvline(50 * wLO * 1e-12, lw = 0.5, color = 'gray', zorder = -666)

    #ax.set_xlim(np.amin(wArr), 5. * wLO * 1e-12)
    ax.set_xlim(np.amin(wArr), np.amax(wArr))
    #ax.set_xlim(80, 90)
    ax.set_ylim(-500, 100.)


    ax.set_xlabel(r"$\omega \, [\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosAsOfFreq" + filename + ".png")

