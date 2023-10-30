import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as patch
import scipy.constants as consts
import scipy.optimize

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


def plotDosAsOfFreq(dos1, dos2, dos3, zArr, L, omegaArr, wLO, wTO, epsInf):
    omegaArr = omegaArr * 1e-12  # convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(omegaArr, dos1[:, 0], color=cmapPink(.3), lw=1., linestyle='', marker='X', markersize=2)
    ax.plot(omegaArr, dos2[:, 0], color=cmapPink(.5), lw=1., linestyle='', marker='X', markersize=2)
    ax.plot(omegaArr, dos3[:, 0], color=cmapPink(.7), lw=1., linestyle='', marker='X', markersize=2)
    ax.plot(omegaArr, dos1[:, 0] + dos2[:, 0] + dos3[:, 0], color=cmapBone(.5), lw=1., linestyle='', marker='x',
            markersize=2, markeredgewidth=.5)

    ax.axvline(wLO * 1e-12, lw=0.5, color='gray')
    ax.axvline(wTO * 1e-12, lw=0.5, color='gray')
    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12))
    ax.axvline(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12, lw=0.5, color='gray')

    ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))

    # ax.set_xticks([np.amin(omegaArr), 0, np.amax(omegaArr)])
    # ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$\omega[\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    # legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosAsOfFreq.png")

def plotDosTotalWithSurfExtra(dos, dosSurf, zArr, L, omegaArr, surfFreqArr, wLO, wTO, epsInf, filename):
    omegaArr = omegaArr * 1e-12  # convert to THz
    surfFreqArr = surfFreqArr * 1e-12  # convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    for zInd, zVal in enumerate(zArr):
        color = cmapPink((zInd + 1.) / (len(zArr) + 17.))
        colorSurf = cmapBone((zInd + 1.) / (len(zArr) + 17.))
        ax.plot(surfFreqArr, dosSurf[:, zInd], color=colorSurf, lw=.8, linestyle='-')
        if(zInd == 0):
            ax.plot(omegaArr, dos[:, zInd], color='black', lw=.8, linestyle='-', zorder = 666)
            continue
        ax.plot(omegaArr, dos[:, zInd], color=color, lw=.8, linestyle='-')

    ax.axvline(wLO * 1e-12, lw=0.5, color='gray')
    ax.axvline(wTO * 1e-12, lw=0.5, color='gray')
#    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12))
#    ax.axvline(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12, lw=0.5, color='gray')

    ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))
    ax.set_xlim(0., 10)
    ax.set_ylim(-1, 5)

    ax.set_xlabel(r"$\omega[\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    # legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosTotal2" + filename + ".png")




def plotDosIntegratedAsOfCutoff(dos, zArr, L, omegaArr, wLO, wTO, epsInf, filename):
    omegaArr = omegaArr * 1e-12  # convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    zArr = zArr[11 : 31 : 3]
    dos = dos[:,11  : 31 : 3]

    for zInd, zVal in enumerate(zArr):
        if(zVal < 1e-9):
            continue
        if(zInd == 0):
            ax.plot(omegaArr, dos[:, zInd], color='black', lw=.6, linestyle='-', zorder = 666,  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
            continue
        color = cmapPink((zInd + 1.) / (len(zArr) + 1.))
        ax.plot(omegaArr, dos[:, zInd], color=color, lw=.8, linestyle='-',  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
    ax.axhline(0., lw = 0.5, color = 'black')
    #ax.axvline(wLO * 1e-12, lw=0.5, color='gray')
    #ax.axvline(wTO * 1e-12, lw=0.5, color='gray')
#    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12))
#    ax.axvline(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12, lw=0.5, color='gray')

    ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))
    #ax.set_xlim(np.amin(omegaArr), 30)
    ax.set_ylim(-10000, 100.)

    ax.set_xlabel(r"$\Lambda[\mathrm{THz}]$")
    ax.set_ylabel(r"$ \langle E^2(r) \rangle_{\Lambda} \, \left[\frac{\mathrm{V}}{\mathrm{m}}\right] $")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.1), edgecolor='black', ncol=3)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosIntegrated" + filename + ".png")

def plotFluctuationsE(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf):

    zArr = zArr[1:]
    dosNoSurf = dosNoSurf[1:]
    dosTot = dosTot[1:]

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.95)
    ax = plt.subplot(gs[0, 0])
    #axRight = ax.twinx()


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
#    ax.plot(zArr, np.abs(dosTot) * 1e-12, color=cmapBone(.6), linestyle='', marker='x', markersize=2., label = r"$\langle E^2 \rangle_{\mathrm{tot}} - \langle E^2 \rangle_{\mathrm{vac}}$")
#    ax.plot(zArr, np.abs(dosNoSurf) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle E^2 \rangle_{\mathrm{no \, surf.}} - \langle E^2 \rangle_{\mathrm{vac}} \right)$")

    #ax.plot(zArr, dosTot * 1e-12, color=cmapBone(.6), linestyle='', marker='x', markersize=2.,
    #        label=r"$\langle E^2 \rangle_{\mathrm{tot}} - \langle E^2 \rangle_{\mathrm{vac}}$")
    ax.plot(zArr, dosNoSurf * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2.,
            label=r"$- \left(\langle E^2 \rangle_{\mathrm{no \, surf.}} - \langle E^2 \rangle_{\mathrm{vac}} \right)$")

    #ax.plot(zArr, np.abs(dosTot[-1]) * 1e-12 * zArr[-1]**3 / zArr**3, color = 'red', lw = 0.5)
    #ax.plot(zArr, np.abs(dosNoSurf[18]) * 1e-12 * zArr[18]**2 / zArr**2, color = 'red', lw = 0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_xlim(np.amin(omegaArr), 30)
    #ax.set_ylim(1e-12, 1e3)

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$ \langle E^2(r) \rangle \, \left[\frac{\mathrm{MV}^2}{\mathrm{m}^2}\right] $")
    #axRight.set_ylabel(r"$ \langle E^2(r) \rangle \, \left[\frac{\mathrm{V}^2}{\mathrm{m}^2}\right] $")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        #axRight.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/FluctuationsEFieldSPhP.png")


def plotFluctuationsA(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf):

    zArr = zArr[1:]
    dosNoSurf = dosNoSurf[1:]
    dosTot = dosTot[1:]

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.95)
    ax = plt.subplot(gs[0, 0])
    #axRight = ax.twinx()


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
#    ax.plot(zArr, np.abs(dosTot) * 1e-12, color=cmapBone(.6), linestyle='', marker='x', markersize=2., label = r"$\langle A^2 \rangle_{\mathrm{tot}} - \langle A^2 \rangle_{\mathrm{vac}}$")
#    ax.plot(zArr, np.abs(dosNoSurf) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle A^2 \rangle_{\mathrm{no \, surf.}} - \langle A^2 \rangle_{\mathrm{vac}} \right)$")
    ax.plot(zArr, dosTot * 1e-12, color=cmapBone(.6), linestyle='', marker='x', markersize=2.,
            label=r"$\langle A^2 \rangle_{\mathrm{tot}} - \langle A^2 \rangle_{\mathrm{vac}}$")
    ax.plot(zArr, dosNoSurf * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2.,
            label=r"$- \left(\langle A^2 \rangle_{\mathrm{no \, surf.}} - \langle A^2 \rangle_{\mathrm{vac}} \right)$")

    #ax.plot(np.log(zArr), np.abs(dosNoSurf) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle E^2 \rangle_{\mathrm{no \, surf.}} - \langle E^2 \rangle_{\mathrm{vac}} \right)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_xlim(np.amin(omegaArr), 30)
    #ax.set_ylim(1e-15, 1e0)

    #ydat = np.abs(dosNoSurf) * 1e-12
    #fitInd = 22
    #logZArr = np.log(zArr)
    #slope = (ydat[fitInd] - ydat[fitInd - 1]) / (logZArr[fitInd] - logZArr[fitInd - 1])
    #offset = ydat[fitInd] - slope * logZArr[fitInd]
    #ax.plot(logZArr, slope * logZArr + offset, color = 'red', lw = 0.5)

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$ \langle A^2(r) \rangle \, \left[\frac{\mathrm{MV}^2}{\mathrm{m}^2} \mathrm{THz}^2\right] $")
    #axRight.set_ylabel(r"$ \langle E^2(r) \rangle \, \left[\frac{\mathrm{V}^2}{\mathrm{m}^2}\right] $")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        #axRight.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/FluctuationsAFieldSPhP.png")



def plotFluctuationsEandA(dosNoSurfE, dosTotE, dosNoSurfA, dosTotA, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3.5, 2.), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                           wspace=0.35, hspace=0., top=0.95, bottom=0.18, left=0.2, right=0.95)
    axE = plt.subplot(gs[0, 0])
    axA = plt.subplot(gs[1, 0])
    #axRight = ax.twinx()


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axE.plot(zArr, np.abs(dosTotE) * 1e-12, color=cmapBone(.5), linestyle='', marker='x', markersize=2., label = r"$\langle A^2 \rangle_{\mathrm{tot}} - \langle A^2 \rangle_{\mathrm{vac}}$")
    axE.plot(zArr, np.abs(dosNoSurfE) * 1e-12, color=cmapPink(.5), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle A^2 \rangle_{\mathrm{no \, surf.}} - \langle A^2 \rangle_{\mathrm{vac}} \right)$")

#    axA.plot(zArr, np.abs(dosTotA) * 1e-12, color=cmapBone(.6), linestyle='', marker='x', markersize=2., label = r"$\langle A^2 \rangle_{\mathrm{tot}} - \langle A^2 \rangle_{\mathrm{vac}}$")
#    axA.plot(zArr, np.abs(dosNoSurfA) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle A^2 \rangle_{\mathrm{no \, surf.}} - \langle A^2 \rangle_{\mathrm{vac}} \right)$")
    axA.plot(zArr, dosTotA, color=cmapBone(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Total}$")
    axA.plot(zArr, dosNoSurfA, color=cmapPink(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{No} \, \mathrm{SPhP}$")

    #ax.plot(np.log(zArr), np.abs(dosNoSurf) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle E^2 \rangle_{\mathrm{no \, surf.}} - \langle E^2 \rangle_{\mathrm{vac}} \right)$")
    axE.set_xscale("log")
    axE.set_yscale("log")
    axA.set_xscale("log")
    #axA.set_yscale("log")
    axE.set_xlim(1e-8, 1e-4)
    axE.set_ylim(1e-12, 1e0)
    axA.set_xlim(1e-8, 1e-4)
    axA.set_ylim(- 11, 12)
    #axA.set_ylim(1e-14, 1e-4)

    axE.plot(zArr[20:], np.abs(dosTotE[-1]) * 1e-12 * zArr[-1]**3 / zArr[20:]**3, color = 'red', lw = 0.3)
    axE.plot(zArr, np.abs(dosNoSurfE[17]) * 1e-12 * zArr[17]**2 / zArr**2, color = 'red', lw = 0.3)

    axA.plot(zArr[:], np.abs(dosTotA[18]) * zArr[18]**3 / zArr[:]**3, color = 'red', lw = 0.3)

    #axE.axvline(consts.c / (3. * 241.8 * 1e12), color = "gray", lw = .5)
    #axA.axvline(consts.c / (3. * 241.8 * 1e12), color = "gray", lw = .5)

    ydat = np.abs(dosNoSurfA)
    fitInd = 29
    logZArr = np.log(zArr)
    slope = (ydat[fitInd] - ydat[fitInd - 1]) / (logZArr[fitInd] - logZArr[fitInd - 1])
    offset = ydat[fitInd] - slope * logZArr[fitInd]
    #axA.plot(np.exp(logZArr), - slope * logZArr + 0. * offset, color = 'red', lw = 0.5)
    axA.plot(zArr, -(slope * logZArr + offset), color = 'red', lw = 0.3)

    #axA.plot(zArr[:], np.abs(dosTotA[18]) * (np.log(zArr[18]) - np.log(zArr[:])), color = 'red', lw = 0.3)


    axA.set_xlabel(r"$z[\mathrm{m}]$", fontsize = 8)
    axE.set_ylabel(r"$ \langle E^2 \rangle_{\rm eff}  \left[\frac{\mathrm{MV}^2}{\mathrm{m}^2}\right] $", fontsize = 8, labelpad = 1)
    axA.set_ylabel(r"$ \langle A^2 \rangle_{\rm eff}  \left[\frac{\mathrm{V}^2}{\mathrm{THz}^2 \, \mathrm{m}^2}\right] $", fontsize = 8, labelpad = 6)
    #axRight.set_ylabel(r"$ \langle E^2(r) \rangle \, \left[\frac{\mathrm{V}^2}{\mathrm{m}^2}\right] $")

    axE.set_xticks([])
    axE.set_yticks([1e-10, 1e-5, 1.])
    axE.set_yticklabels(["$10^{-10}$", "$10^{-5}$", "$1$"], fontsize = 8)

    axA.set_yticks([-10, 0, 10])
    axA.set_yticklabels(["$-10$", "$0$", "$10$"], fontsize = 8)


    axE.text(1.5 * 1e-5, 1e-9, r"$\sim z^{-2}$", fontsize = 7, color = "red")
    axE.text(1.5 * 1e-6, 1e-6, r"$\sim z^{-3}$", fontsize = 7, color = "red")

    axA.text(1. * 1e-5, 8, r"$\sim z^{-3}$", fontsize = 7, color = "red")
    axA.text(5. * 1e-7, -7, r"$\sim \log(z) + \mathrm{const}$", fontsize = 7, color = "red")


    legend = axA.legend(fontsize=8, loc='upper left', bbox_to_anchor=(.0, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)
        axA.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/FluctuationsEandASPhP.png")



def plotFluctuationsENaturalUnits(dosNoSurfE, dosTotE, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(7.04 / 3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.95, bottom=0.22, left=0.28, right=0.95)
    axE = plt.subplot(gs[0, 0])

    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    lambda0 = consts.c / (2. * np.pi * wInf)
    natUnitFac = consts.epsilon_0 * lambda0**3 / (consts.hbar * wInf)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axE.plot(zArr / lambda0, np.abs(dosTotE) * natUnitFac, color=cmapBone(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Total}$")
    axE.plot(zArr / lambda0, np.abs(dosNoSurfE) * natUnitFac, color=cmapPink(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Bulk}$")

    axE.set_xscale("log")
    axE.set_yscale("log")
    axE.set_xlim(1e-2, 1e2)
    axE.set_ylim(1e-6, 1e3)

    axE.plot(zArr / lambda0, np.abs(dosTotE[-1]) * zArr[-1]**3 / zArr**3 * natUnitFac, color = 'red', lw = 0.3)
    axE.plot(zArr / lambda0, np.abs(dosNoSurfE[20]) * zArr[20]**2 / zArr**2 * natUnitFac, color = 'red', lw = 0.3)

    axE.set_xlabel(r"$z[\lambda_0]$", fontsize = 8)
    axE.set_ylabel(r"$ \langle E^2 \rangle_{\rm eff}  \left[\frac{\hbar \omega_{\mathrm{S}}}{\varepsilon_0 \lambda_0^3}\right] $", fontsize = 8, labelpad = 1)

    #axE.set_xticks([])
    #axE.set_yticks([1e-10, 1e-5, 1.])
    #axE.set_yticklabels(["$10^{-10}$", "$10^{-5}$", "$1$"], fontsize = 8)

    axE.text(1e-1, 1e1, r"$\sim z^{-3}$", fontsize = 7, color = "red")
    axE.text(1.5 * 1e-2, 5. * 1e0, r"$\sim z^{-2}$", fontsize = 7, color = "red")

    legend = axE.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/FluctuationsENatUnits.png")



def plotFluctuationsWithFit(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf, filename):

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.95)
    ax = plt.subplot(gs[0, 0])
    #axRight = ax.twinx()


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    #ax.plot(zArr, np.abs(dosTot) * 1e-12, color=cmapBone(.6), linestyle='', marker='x', markersize=2., label = r"$\langle E^2 \rangle_{\mathrm{tot}} - \langle E^2 \rangle_{\mathrm{vac}}$")
    #axRight.plot(zArr, dosNoSurf, color=cmapPink(.6), linestyle='', marker='x', markersize=2.)
    #ax.plot(zArr, np.abs(dosNoSurf) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle E^2 \rangle_{\mathrm{no \, surf.}} - \langle E^2 \rangle_{\mathrm{vac}} \right)$")
    ax.plot(np.log(zArr), np.abs(dosNoSurf) * 1e-12, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$- \left(\langle E^2 \rangle_{\mathrm{no \, surf.}} - \langle E^2 \rangle_{\mathrm{vac}} \right)$")
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_xlim(np.amin(omegaArr), 30)
    #ax.set_ylim(1e-14, 1e-12)

    ydat = np.abs(dosNoSurf) * 1e-12

    #ax.plot(zArr, np.abs(dosTot[-1]) * 1e-12 * zArr[-1]**3 / zArr**3, color = 'red', lw = 0.5)
    fitInd = 22
    #ax.plot(zArr, np.abs(dosNoSurf[fitInd]) * 1e-12 * zArr[fitInd] / zArr, color = 'red', lw = 0.5)
    #ax.plot(zArr[fitInd:], np.abs(dosNoSurf[fitInd]) * 1e-12 / (np.log(1. / zArr[fitInd]) - np.log(1. / zArr[fitInd-1])) * (np.log(1. / zArr[fitInd:]) - np.log(1. / zArr[fitInd])), color = 'red', lw = 0.5)
    logZArr = np.log(zArr)
    slope = (ydat[fitInd] - ydat[fitInd - 1]) / (logZArr[fitInd] - logZArr[fitInd - 1])
    offset = ydat[fitInd] - slope * logZArr[fitInd]
    ax.plot(logZArr, slope * logZArr + offset, color = 'red', lw = 0.5)


    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$ \langle E^2(r) \rangle \, \left[\frac{\mathrm{MV}^2}{\mathrm{m}^2}\right] $")
    #axRight.set_ylabel(r"$ \langle E^2(r) \rangle \, \left[\frac{\mathrm{V}^2}{\mathrm{m}^2}\right] $")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        #axRight.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosIntegrated" + filename + ".png")

def plotEffectiveMass(delMOverM, zArr, L, wLO, wTO, epsInf, filename):

    fig = plt.figure(figsize=(3, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.95)
    ax = plt.subplot(gs[0, 0])
    #axRight = ax.twinx()


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(zArr, delMOverM, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$\mathrm{all}$")
    #ax.axhline(1., lw = 0.3, color = 'red')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-8, 1e-5)
    ax.set_ylim(1e-8, 1.)
    #ax.set_xlim(np.amin(omegaArr), 30)
    #ax.set_ylim(1e-14, 1e-12)

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$ \frac{\Delta m}{m} $")

    #legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        #axRight.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosIntegrated" + filename + ".png")



def plotDosIntegratedHopping(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf, filename):

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.85)
    ax = plt.subplot(gs[0, 0])
    axRight = ax.twinx()

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(zArr, -dosTot, color=cmapBone(.6), linestyle='', marker='x', markersize=2.)
    axRight.plot(zArr, -dosNoSurf, color=cmapPink(.6), linestyle='', marker='x', markersize=2.)
    ax.set_xscale("log")
    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_xlim(np.amin(omegaArr), 30)
    ax.set_ylim(-0.15, 0.)

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$ \frac {\Delta t}{t_0} $")
    axRight.set_ylabel(r"$ \frac {\Delta t}{t_0} $")

    #legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.1), edgecolor='black', ncol=3)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        axRight.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosIntegrated" + filename + ".png")




def compareSPhPInt(dosAna, dosNum, zArr, filename):
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(zArr, dosAna, color = cmapPink(.5), lw=.8, linestyle = '-')
    ax.plot(zArr, dosNum, color = cmapBone(.5), lw=.8, linestyle = '--')

    #ax.axhline(1., lw = .5, color = 'gray')
    #ax.axhline(0.01, lw = .5, color = 'gray')

    #ax.axvline(1e-9, lw = .5, color = 'gray')
    #ax.axvline(1e-8, lw = .5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    #ax.set_ylabel(r"$\rho / \rho_0$")
    ax.set_ylabel(r"$\langle A^2(r) \rangle \, \left[\frac{\mathrm{V}^2}{\mathrm{m}^2 \mathrm{THz}^2} \right]$")

    ax.set_xscale('log')
    ax.set_yscale('log')

    # legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosTotal" + filename + ".png")