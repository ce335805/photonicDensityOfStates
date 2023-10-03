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

def plotDosWhole(zArr, wLO, wTO, epsInf, L):

    arrBelow, arrWithin, arrAbove, surfFreqArr = prod.defineFreqArrays(wLO, wTO, epsInf)
    wArr = np.append(arrBelow, arrWithin)
    wArr = np.append(wArr, arrAbove)

    dosTETotal = prod.retrieveDosTE(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)
    dosTMTotal = prod.retrieveDosTMPara(arrBelow[-1], arrWithin[-1], arrAbove[-1], L, epsInf)

    filename = "TMPara"
    createDosPlotFreq(wArr, zArr, dosTMTotal + dosTETotal, filename, wLO, wTO)

def createDosPlotFreq(wArr, zArr, dos, filename, wLO, wTO):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    indArr = np.array([0, 5, 10, 15, 20], dtype = int)
    print(dos[:, indArr[0]])
    ax.plot(wArr * 1e-12, dos[:, indArr[0]], color=cmapPink(0.1), lw=.7, label = "$z = $" + "{:1.2g}".format(zArr[indArr[0]]))
    ax.plot(wArr * 1e-12, dos[:, indArr[1]], color=cmapPink(0.3), lw=.7, label = "$z = $" + "{:1.2g}".format(zArr[indArr[1]]))
    ax.plot(wArr * 1e-12, dos[:, indArr[2]], color=cmapPink(0.5), lw=.7, label = "$z = $" + "{:1.2g}".format(zArr[indArr[2]]))
    ax.plot(wArr * 1e-12, dos[:, indArr[3]], color=cmapPink(0.6), lw=.7, label = "$z = $" + "{:1.2g}".format(zArr[indArr[3]]))
    ax.plot(wArr * 1e-12, dos[:, indArr[4]], color=cmapPink(0.7), lw=.7, label = "$z = $" + "{:1.2g}".format(zArr[indArr[4]]))


    ax.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    ax.set_xlim(np.amin(wArr * 1e-12), 2. * wLO * 1e-12)
    ax.set_ylim(0., 10.)


    ax.set_xlabel(r"$\omega \, [\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.0), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosAsOfFreq" + filename + ".png")

