import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors

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

    dos = dos[:, :] - (dos[:, 0])[:, None] * np.ones(50)[None, :]

    for zInd, zVal in enumerate(zArr):
        if(zVal < 1e-9):
            continue
        if(zInd == 0):
            ax.plot(omegaArr, dos[:, zInd] * omegaArr**3, color='black', lw=.6, linestyle='-', zorder = 666,  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
            continue
        #if(zInd < 15 or zInd > 18):
        #    continue
        color = cmapPink((zInd + 1.) / (len(zArr) + 1.))
        if(zInd % 5 == 0):
            ax.plot(omegaArr, dos[:, zInd] * omegaArr**3, color=color, lw=.8, linestyle='-',  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
        else:
            ax.plot(omegaArr, dos[:, zInd] * omegaArr**3, color=color, lw=.8, linestyle='-')
    #ax.axvline(wLO * 1e-12, lw=0.5, color='gray')
    #ax.axvline(wTO * 1e-12, lw=0.5, color='gray')
#    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12))
#    ax.axvline(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12, lw=0.5, color='gray')

    ax.axhline(0., color = 'blue', lw = .3, zorder = 1000)

    #ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))
    ax.set_xlim(np.amin(omegaArr), 30)
    ax.set_ylim(-5, 5.)

    ax.set_xlabel(r"$\omega[\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=2)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

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

    plt.savefig("./SPhPPlotsSaved/dosTotal" + filename + ".png")




def plotDosIntegratedAsOfCutoff(dos, zArr, L, omegaArr, wLO, wTO, epsInf, filename):
    omegaArr = omegaArr * 1e-12  # convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')


    for zInd, zVal in enumerate(zArr):
        if(zVal < 1e-9):
            continue
        if(zInd == 0):
            ax.plot(omegaArr, dos[:, zInd], color='black', lw=.6, linestyle='-', zorder = 666,  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
            continue
        #if(zInd > 20):
        #    continue
        color = cmapPink((zInd + 1.) / (len(zArr) + 1.))
        if(zInd % 5 == 0):
            ax.plot(omegaArr, dos[:, zInd], color=color, lw=.8, linestyle='-',  label = r'$z = $'+'{0:.1g}'.format(zVal) + '$\mathrm{m}$')
        else:
            ax.plot(omegaArr, dos[:, zInd], color=color, lw=.8, linestyle='-')
    ax.axvline(wLO * 1e-12, lw=0.5, color='gray')
    ax.axvline(wTO * 1e-12, lw=0.5, color='gray')
#    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12))
#    ax.axvline(np.sqrt((epsInf * wLO ** 2 - wTO ** 2) / (epsInf - 1)) * 1e-12, lw=0.5, color='gray')

    #ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))
    ax.set_xlim(np.amin(omegaArr), 30)
    ax.set_ylim(-100, 20)

    ax.set_xlabel(r"$\omega[\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=2)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("./SPhPPlotsSaved/dosIntegrated" + filename + ".png")


def compareSPhPInt(dosAna, dosNum, zArr, filename):
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(zArr, dosAna, color = cmapPink(.5), lw=.8)
    ax.plot(zArr, dosNum, color = cmapBone(.5), lw=.8, linestyle = '--')

    ax.axhline(1., lw = .5, color = 'gray')
    ax.axhline(0.01, lw = .5, color = 'gray')

    ax.axvline(1e-9, lw = .5, color = 'gray')
    ax.axvline(1e-8, lw = .5, color = 'gray')

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    ax.set_xscale('log')
    ax.set_yscale('log')

    # legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosTotal" + filename + ".png")