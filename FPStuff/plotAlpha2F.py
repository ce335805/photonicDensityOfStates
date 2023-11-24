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


def plotLinesAsOfD(qArr, dArr, alpha2Trans, alpha2Long):

    fig = plt.figure(figsize=(2.5, 2.), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                           wspace=0.35, hspace=0., top=0.9, bottom=0.2, left=0.22, right=0.95)
    axLong = plt.subplot(gs[0, 0])
    axTrans = plt.subplot(gs[1, 0])

    #alpha2TransEff = alpha2Trans[:, :] - alpha2Trans[-1, :]
    #alpha2LongEff = alpha2Long[:, :] - alpha2Long[-1, :]

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    linew = 0.8
    for dInd, dVal in enumerate(dArr):
        revInd = len(dArr) - dInd - 1
        if(dInd == 0):
            colorB = cmapBone(.7)
            colorP = cmapPink(.7)
            axLong.plot(qArr * 1e-6, alpha2Long[revInd, :], color=colorB, lw=linew, label="$d = 1\mathrm{cm}$")
            axTrans.plot(qArr * 1e-6, alpha2Trans[revInd, :], color=colorP, lw=linew, label="$d = 1\mathrm{cm}$")
        if (dInd == 1):
            colorB = cmapBone(.5)
            colorP = cmapPink(.5)
            axLong.plot(qArr * 1e-6, alpha2Long[revInd, :], color=colorB, lw=linew, label="$d = 100\mu\mathrm{m}$")
            axTrans.plot(qArr * 1e-6, alpha2Trans[revInd, :], color=colorP, lw=linew, label="$d = 100\mu\mathrm{m}$")
        if (dInd == 2):
            colorB = cmapBone(.2)
            colorP = cmapPink(.2)
            axLong.plot(qArr * 1e-6, alpha2Long[revInd, :], color=colorB, lw=linew, label="$d = 10\mu\mathrm{m}$")
            axTrans.plot(qArr * 1e-6, alpha2Trans[revInd, :], color=colorP, lw=linew, label="$d = 10\mu\mathrm{m}$")


    axLong.set_ylim(0., 8)
    axTrans.set_ylim(0., 8)

    axLong.set_xlim(0., np.amax(qArr * 1e-6))
    axTrans.set_xlim(0., np.amax(qArr * 1e-6))

    axLong.set_xticks([])

    #ax.plot(qArr, 2. * alpha2TransEff[-1, 0] * qArr[0] / qArr, color = 'red', lw = 0.5)
    #ax.plot(qArr, 2. * alpha2TransEff[-1, -1] * qArr[-1]**2 / qArr**2, color = 'red', lw = 0.5)

    #ax.set_yscale('log')
    #ax.set_xscale('log')

    #ax.set_xlim(0., 0.1 * np.amax(qArr))
    #limBot = -1 * 1e-37
    #limTop = np.amax(alpha2TransEff) - 0.5 * limBot
    #ax.set_ylim(limBot, limTop)

    axTrans.set_xlabel(r"$q[\frac{1}{\mu \mathrm{m}}]$")
    axTrans.set_ylabel(r"$\mathrm{Longitudinal}$", fontsize = 8)
    axLong.set_ylabel(r"$\mathrm{Transversal}$", fontsize = 8)

    axLong.text(np.pi / 20, 8.2, r"$\alpha^2F \, [E_{\mathrm{F}} \mathrm{\AA}^2]$", fontsize=8, ha='center')

    legend = axLong.legend(fontsize=8, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    legend = axTrans.legend(fontsize=8, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axTrans.spines[axis].set_linewidth(.5)
        axLong.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/alpha2F.png")