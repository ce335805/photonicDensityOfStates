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

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
    filenameSPhP = "./savedData/massThesisFinal" + wLOStr + wTOStr + ".hdf5"
    cutoff2, zArrSPP, massSPPArr = prodV2.readMasses(filenameSPhP)
    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf
    unitFac1 = consts.m_e / (consts.hbar * wLO / consts.c**2)
    print(unitFac1)

    lambda30 = 2. * np.pi * consts.c / wLO

    wLO = 50 * 1e12
    wTO = 1e6
    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPhP = "./savedData/massThesisFinal" + wLOStr + wTOStr + ".hdf5"
    cutoff2, zArrSPP2, massSPPArr2 = prodV2.readMasses(filenameSPhP)
    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda1 = 2. * np.pi * consts.c / wInf
    unitFac2 = consts.m_e / (consts.hbar * wLO / consts.c**2)

    print(len(zArrSPP))
    print(len(zArrSPP2))

    fig = plt.figure(figsize=(6.5, 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1],
                           wspace=0.4, hspace=0.6, top=0.9, bottom=0.25, left=0.1, right=0.98)

    axIm = plt.subplot(gs[0, 0])
    axZoom = plt.subplot(gs[0, 1])
    axCollapse = plt.subplot(gs[0, 2])

    lambda50 = 2. * np.pi * consts.c / wLO

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axZoom.plot(zArr / lambda30, massSPhPArr, color = cmapPink(.45), lw = 0.8, label = r"$\mathrm{SPhP}$")
    axZoom.plot(zArrSPP / lambda30, massSPPArr, color = cmapBone(.45), lw = 0.8, label = r"$\mathrm{SPP}$")
    axZoom.axhline(0., color = "black", lw = 0.4)
    #axCollapse.plot(zArrSPP / lambda0, np.abs(massSPPArr) * unitFac1, color = cmapBone(.25), lw = 0.8, label = r"$\omega_{\rm p} = 32 \mathrm{THz}$")
    axCollapse.plot(zArrSPP2 / lambda50, np.abs(massSPPArr2) * unitFac2, color = cmapBone(.35), lw = 0.8, label = r"$\mathrm{SPP}$")
    #axCollapse.plot(zArr / lambda0, np.abs(massSPhPArr), color = cmapPink(.45), lw = 0.8)

    axCollapse.plot(zArrSPP2 / lambda50, np.abs(massSPPArr2[-1]) * zArrSPP2[-1]**3 / zArrSPP2**3 * unitFac2, color = "red", lw = .4)
    fitInd = 150
    axCollapse.plot(zArrSPP2 / lambda50, np.abs(massSPPArr2[fitInd]) * zArrSPP2[fitInd]**1 / zArrSPP2**1 * unitFac2, color = "red", lw = .4)

    axCollapse.set_xscale("log")
    axCollapse.set_yscale("log")

    axZoom.set_xlim(0., 10.)
    axZoom.set_ylim(- .5 * 1e-8, .5 * 1e-8)

    axCollapse.set_xlim(2. * 1e-3, 1e1)
    axCollapse.set_ylim(2. * 1e-3, 1e5)

    axZoom.set_xlabel("$z [\lambda^*]$", fontsize = 8)
    axCollapse.set_xlabel("$z [\lambda^*]$", fontsize = 8)

    axZoom.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad = 6., fontsize = 8)
    axCollapse.set_ylabel(r"$|\Delta m| \, \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \omega_{\rm p}}{c^2} \right]$", labelpad = 6., fontsize = 8)

    axZoom.set_xticks([0., 1., 10.])
    axZoom.set_xticklabels(["$0$", "$1$", "$10$"], fontsize = 8)

    #axSPhP.set_yticks([0., 1.])
    #axSPhP.set_yticklabels(["$0$", "$1$"], fontsize = 8)

    axCollapse.text(0.25, 0.8, r"$\sim z^{-3}$", transform = axCollapse.transAxes)
    axCollapse.text(0.72, 0.25, r"$\sim z^{-1}$", transform = axCollapse.transAxes)

    legend = axZoom.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    legend = axCollapse.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)


    axIm.set_xticks([])
    axIm.set_yticks([])
    arr_img = mpimg.imread('./ThesisPlots/pngMassComic.png')
    im = OffsetImage(arr_img, zoom = .3)
    ab = AnnotationBbox(im, (-0.28, -0.2), xycoords='axes fraction', box_alignment=(0, 0), frameon=False, pad=0, boxcoords="offset points")
    axIm.add_artist(ab)

    axIm.text(-0.4, 1.04, r"$\mathrm{(a)}$", fontsize = 8, transform = axIm.transAxes)
    axZoom.text(-0.37, 1.04, r"$\mathrm{(b)}$", fontsize = 8, transform = axZoom.transAxes)
    axCollapse.text(-0.34, 1.04, r"$\mathrm{(c)}$", fontsize = 8, transform = axCollapse.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axZoom.spines[axis].set_linewidth(.5)
        axCollapse.spines[axis].set_linewidth(.5)
        axIm.spines[axis].set_linewidth(0.)


    plt.savefig("./ThesisPlots/MassesZoom.png")

