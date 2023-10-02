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


def plotDosAsOfFreqDosTotal(dos, zArr, L, omegaArr, wLO, wTO, epsInf, filename):
    omegaArr = omegaArr * 1e-12  # convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    dos = dos[:, :] - (dos[:, 0])[:, None] * np.ones(len(zArr))[None, :]
    zArr = zArr[ : 21 : 4]
    dos = dos[:,  : 21 : 4]

    for zInd, zVal in enumerate(zArr):
        if(zInd == 0):
            ax.plot(omegaArr, dos[:, zInd], color='black', lw=.6, linestyle='-', zorder = 666,  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
            continue
        color = cmapPink((zInd + 1.) / (len(zArr) + 4.))
        ax.plot(omegaArr, dos[:, zInd] * omegaArr**3, color=color, lw=.6, linestyle='-', label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
        #arrow = patch.FancyArrow(0., 0.3 - 0.05 * zInd, consts.c / zVal * 1e-12, 0., color = color, width = 0.002, head_width = 0.01, head_length=0.3)
        #ax.add_patch(arrow)
    #ax.axvline(wLO * 1e-12, lw=0.3, color='gray')
    #ax.axvline(wTO * 1e-12, lw=0.3, color='gray')
#    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12))
#    ax.axvline(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12, lw=0.5, color='gray')

    ax.axhline(0., color = 'black', lw = .3, zorder = 1000)

    #ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))
    ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))
    #ax.set_ylim(-14.2, 14.2)

    ax.set_xlabel(r"$\omega[\mathrm{THz}]$")
    ax.set_ylabel(r"$\left(\rho / \rho_0 - 0.5\right) \times \omega^3$")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=2)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosTotal" + filename + ".png")



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

def plotDosIntegratedFixedCutoff(dosNoSurf, dosTot, zArr, L, wLO, wTO, epsInf, filename):

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

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.2, right=0.95)
    ax = plt.subplot(gs[0, 0])
    #axRight = ax.twinx()


    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(zArr, delMOverM, color=cmapPink(.6), linestyle='', marker='x', markersize=2., label = r"$\mathrm{all}$")
    ax.axhline(1., lw = 0.3, color = 'red')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    #ax.set_xlim(np.amin(omegaArr), 30)
    #ax.set_ylim(1e-14, 1e-12)

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$ \frac{\Delta m}{m} $")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

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