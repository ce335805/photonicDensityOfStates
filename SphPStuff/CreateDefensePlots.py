import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib.colors import LinearSegmentedColormap
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

def getDosFP():

    dir = "./savedData/DefenceData/"

    fileName = dir + "DosFP.h5"
    h5f = h5py.File(fileName, 'r')
    d = h5f['d'][:]
    wArr = h5f['wArr'][:]
    dos = h5f['dos'][:]
    h5f.close()

    return (d, wArr, dos)

def FabryPerotPlot():
    dFP, wArrFP, dosFP = getDosFP()

    fig = plt.figure(figsize=(4., 2.), dpi=400)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.0, hspace=0.5, top=1., bottom=0.2, left=0.1, right=1.)
    axDos = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    axDos.plot(wArrFP, dosFP, color=cmapBone(0.4), lw=1.2)
    axDos.axhline(1., lw=0.5, color='gray', zorder=-666)

    axDos.set_xlim(0., 1.9 * 1e14)
    axDos.set_ylim(0., 3.)
    w0 = np.pi * consts.c / dFP[0]
    axDos.set_xticks([0., w0, 5. * w0, 9. * w0, 13. * w0, 17. * w0])
    axDos.set_xticklabels([r"$0$", r"$\omega_0$", r"$5\omega_0$", r"$9\omega_0$", r"$13\omega_0$", r"$17\omega_0$"])
    axDos.set_yticks([0., 1, 2])
    axDos.set_yticklabels([r"$0$", r"$1$", r"$2$"], fontsize=8)

    axDos.set_xlabel(r"$\omega_{\rm phot}$")
    axDos.set_ylabel(r"$\rho_{\rm FP} / \rho_0$")

    for axis in ['top', 'bottom', 'left', 'right']:
        axDos.spines[axis].set_linewidth(.5)

    for axis in ['top', 'right']:
        axDos.spines[axis].set_linewidth(.0)

    plt.savefig("../SphPStuff/DefencePlots/FullDOSFP.png")


def envelopeFunc(xArr):

    env = 1. / (1. + np.exp(-8. * (xArr - .6)))


    addAtZero = (1. - 1. / (1. + np.exp(-8. * (0 - .6)))) * np.exp(-200. * xArr)

    return env + addAtZero

def makeComicDos(xArr):

    const = envelopeFunc(xArr)

    peak1 = 0.013 *  0.005 / (0.005**2 + (xArr - 0.1)**2)
    peak2 = 0.01 *  0.005 / (0.005**2 + (xArr - 0.2)**2)
    peak3 = 0.008 *  0.005 / (0.005**2 + (xArr - 0.3)**2)
    peak4 = 0.006 *  0.005 / (0.005**2 + (xArr - 0.4)**2)
    peak5 = 0.004 *  0.005 / (0.005**2 + (xArr - 0.5)**2)
    peak6 = 0.003 *  0.005 / (0.005**2 + (xArr - 0.6)**2)
    peak7 = 0.002 *  0.005 / (0.005**2 + (xArr - 0.7)**2)
    peak8 = 0.0014 *  0.005 / (0.005**2 + (xArr - 0.8)**2)
    peak9 = 0.0008 *  0.005 / (0.005**2 + (xArr - 0.9)**2)
    peak10 = 0.0005 *  0.005 / (0.005**2 + (xArr - 1.)**2)

    yArr = const + peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7 + peak8 + peak9 + peak10

    return yArr


def plotComicDos():
    fig = plt.figure(figsize=(5., 2.), dpi=400)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.0, hspace=0.5, top=1., bottom=0.2, left=0.1, right=1.)
    axDos = plt.subplot(gs[0, 0])

    xArr = np.linspace(0, 1, 1000)
    yArr = makeComicDos(xArr)


    cmapBone = cm.get_cmap('bone')
    plt.plot(xArr, yArr, lw = 1.2, color=cmapBone(0.35))

    axDos.set_xlim(0, 1)
    axDos.set_ylim(0, 3)

    axDos.set_yticks([0, 1])
    axDos.set_xticks([])

    axDos.set_xlabel(r"$\omega$", fontsize = 10)
    axDos.set_ylabel(r"$\rho / \rho_0$", fontsize = 10)
    axDos.axhline(1, color = 'black', lw = 0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        axDos.spines[axis].set_linewidth(.5)

    for axis in ['top', 'right']:
        axDos.spines[axis].set_linewidth(.0)


    plt.savefig("../SphPStuff/DefencePlots/Comicdos.png")


def plotQuadDispersion():

    xArr = np.linspace(-1., 1., 1000, endpoint = True)
    yArr = xArr ** 2

    fig = plt.figure(figsize=(3.2, 2.2), dpi=300)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=1., bottom=0., left=0.0, right=.95)

    ax = plt.subplot(gs[0, 0])

    #ax.plot(xArr, 0.7 * yArr, color = 'red', lw = 1.5, linestyle = "--", label = r"$\varepsilon_{\rm eff}$", zorder = 666)
    ax.plot(xArr, yArr, color = 'black', lw = 1.5, label = r"$\frac{p^2}{2m}$")


    ax.set_xticks([])
    ax.set_yticks([])


    ax.arrow(0., -0.1, 0., 1.1, length_includes_head=True,
              head_width=0.04, head_length=0.04, linewidth = 1., color = "black")
    ax.arrow(-1., 0., 2., 0., length_includes_head=True,
              head_width=0.04, head_length=0.04, linewidth = 1., color = "black")

    ax.text(1.03, -0.03, r"$p$", fontsize = 16)
    #ax.text(0.05, 0.85, r"$\varepsilon(k)$", fontsize = 20)

    legend = ax.legend(fontsize=25, loc='upper right', bbox_to_anchor=(0.87, 0.98), edgecolor='black', ncol=1,
                       handlelength=0.8, handletextpad=0.2, labelspacing = 0.)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.0)

    plt.savefig("./DefencePlots/FreeDispersion.png")



def main():
    #FabryPerotPlot()
    plotQuadDispersion()
    #plotComicDos()

main()