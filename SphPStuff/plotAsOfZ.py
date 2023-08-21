import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
from matplotlib import gridspec

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

def plotDosAsOfFreqDosTotal(dos, zArr, L, omegaArr, wLO, wTO, epsInf, filename):
    omegaArr = omegaArr * 1e-12  # convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    for wInd, wVal in enumerate(omegaArr):
        if(wInd == len(omegaArr) - 3):
            continue
        color = cmapPink((wInd + 1.) / (len(omegaArr) + 2.))
        ax.plot(zArr, dos[wInd, :], color=color, lw=.8, linestyle='-', label = "$\omega = $" + "{0:.2g}".format(wVal) + r"$\mathrm{THz}$")

    ax.axvline(0., lw=0.5, color='gray')
    ax.axhline(0.5, lw=0.5, color='gray')
    ax.axhline(0.5 * np.sqrt(2.), lw=0.5, color='gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_ylim(-0.3, 5)

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, .97), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosTotalAsOfZDielectric" + filename + ".png")