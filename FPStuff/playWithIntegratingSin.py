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

def integrateSin():

    xArr = np.linspace(0, 1e4, int(1e5))
    yArr = np.sin(xArr) * xArr**2

    print("starting computations")

    alphaArr = np.logspace(-2.7, 1, 30)
    expFac = np.exp(-alphaArr[:, None] * xArr[None, :])
    integrand = yArr[None, :] * expFac[:, :]

    ind = 5
    print("alpha(ind) = {}".format(alphaArr[ind]))
    plotSinWithDecay(xArr, integrand[ind, :])

    res = np.trapz(integrand, x = xArr, axis=1)
    plotResAsOfAlpha(alphaArr, res)

def integrateSinAlt():
    print("starting computations")

    cutoffArr = np.logspace(0, 3, 50) * 2. * np.pi
    N = int(1e6)
    xArrs = np.zeros((len(cutoffArr), N))
    for cutInd, cutVal in enumerate(cutoffArr):
        xArrs[cutInd, :] = np.linspace(0, cutVal, N)
    yArr = np.sin(xArrs) * xArrs**2

    #alphaArr = 1. / cutoffArr * 7.
    alphaArr = 1. / cutoffArr * 22.5
    expFac = np.exp(-alphaArr[:, None] * xArrs[:, :])
    integrand = yArr[:, :] * expFac[:, :]
    #ind = 5
    #print("alpha(ind) = {}".format(alphaArr[ind]))
    #plotSinWithDecay(xArr, integrand[ind, :])

    res = np.zeros(len(alphaArr))
    for cutInd, cutVal in enumerate(cutoffArr):
        res[cutInd] = np.trapz(integrand[cutInd, :], x = xArrs[cutInd, :])
    print(cutoffArr)
    plotResAsOfAlpha(cutoffArr, res)

def plotResAsOfAlpha(alphaArr, res):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(alphaArr, res, color=cmapPink(.5), linestyle = '', marker = 'x', markersize = 2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\int \sin$")
    ax.set_xscale('log')
    ax.axhline(1., color = 'black', lw = 0.5)

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./FPPlotsSaved/sinIntegralAlpha.png")

def plotSinWithDecay(xArr, yArr):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(xArr, yArr, color=cmapPink(.5), linestyle = '-')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\langle E^2 \rangle$")
    #ax.set_xscale('log')

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./FPPlotsSaved/sinWithDecay.png")

#integrateSin()
integrateSinAlt()


