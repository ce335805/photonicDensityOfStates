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

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    alpha2TransEff = alpha2Trans[:, :] - alpha2Trans[-1, :]
    alpha2LongEff = alpha2Long[:, :] - alpha2Long[-1, :]

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    for dInd, dVal in enumerate(dArr):
        colorP = cmapPink((dInd + 1.) / (len(dArr) + 2))
        colorB = cmapBone((dInd + 1.) / (len(dArr) + 2))
        ax.plot(qArr, alpha2TransEff[dInd, :], color = colorP, lw = 0.5)
        ax.plot(qArr, alpha2LongEff[dInd, :], color = colorB, lw = 0.5)

    #ax.set_xlim(0., 0.1 * np.amax(qArr))
    #limBot = -1 * 1e-37
    #limTop = np.amax(alpha2TransEff) - 0.5 * limBot
    #ax.set_ylim(limBot, limTop)

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\alpha^2F / \rho^2 \, [\mathrm{eV} \mathrm{m}^4]$")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/alpha2F.png")