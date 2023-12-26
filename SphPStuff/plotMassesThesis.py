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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches

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

def plotEffectiveMassesComparison():

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPhP = "./savedData/massThesis1ForPlotting" + wLOStr + wTOStr + ".hdf5"
    cutoff, zArr, massSPhPArr = prodV2.readMasses(filenameSPhP)

    wTO = 1e6
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPhP = "./savedData/massThesis1ForPlotting" + wLOStr + wTOStr + ".hdf5"
    cutoff2, zArrSPP, massSPPArr = prodV2.readMasses(filenameSPhP)

    assert(cutoff == cutoff2)
    #assert(all(zArr == zArr2))

    fig = plt.figure(figsize=(5., 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1],
                           wspace=0.4, hspace=0.6, top=0.75, bottom=0.25, left=0.1, right=0.98)
    axZoom = plt.subplot(gs[0, 0])
    axCollapse = plt.subplot(gs[0, 1])

    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axZoom.plot(zArr, massSPhPArr, color = cmapPink(.45), lw = 0.8)
    axZoom.plot(zArrSPP, massSPPArr, color = cmapBone(.45), lw = 0.8)

    #axSPhP.set_xscale("log")
    #axSPhP.set_yscale("log")

    axZoom.set_xlim(1e-6, 2. * 1e-3)
    axZoom.set_ylim(- 1e-9, 1e-9)

    axZoom.set_xlabel("$z [\lambda_0]$", fontsize = 8)

    axZoom.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad = 6., fontsize = 8)

    #axSPhP.set_xticks([1e-2, 1, 1e2])
    #axSPhP.set_xticklabels(["$10^{-2}$", "$1$", "$10^{2}$"], fontsize = 8)

    #axSPhP.set_yticks([0., 1.])
    #axSPhP.set_yticklabels(["$0$", "$1$"], fontsize = 8)

    for axis in ['top', 'bottom', 'left', 'right']:
        axZoom.spines[axis].set_linewidth(.5)
        axCollapse.spines[axis].set_linewidth(.5)


    plt.savefig("./ThesisPlots/MassesZoom.png")

