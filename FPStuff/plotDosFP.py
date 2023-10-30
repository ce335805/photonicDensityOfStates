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

import handleIntegralData

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

def plotDosAsOfZ(zArr, dosFP, omega, L):

    rho0 = omega**2 / np.pi**2 / 3. / consts.c**3

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(zArr, dosFP / rho0, color='indianred', lw=1.)
    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/dosFPAsOfZ.png")

def plotDosAsOfZSeveralOm(dosFP, zArr, omArr, L):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.8, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    for omInd, omVal in enumerate(omArr):
        color = cmapPink(omInd / (omArr.shape[0] + 0.5))
        rho0 = omVal ** 2 / np.pi ** 2 / 3. / consts.c ** 3
        ax.plot(zArr * 1e6, dosFP[:, omInd] / rho0, color=color, lw=1., label = r"{}".format(int(omVal * 1e-12)) + r"$\rm{THz}$")

    ax.set_xlim(np.amin(zArr) * 1e6, np.amax(zArr) * 1e6)
    ax.axhline(1., color = 'black', lw = 0.5)

    ax.set_xlabel(r"$z \, [\mu \rm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    legend = ax.legend(fontsize=fontsize, loc='lower center', bbox_to_anchor=(0.5, 0.95), edgecolor='black', ncol=2)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig("FPPlotsSaved/dosFPAsOfZOmegas.png")


def plotDosAsOfOm(omArr, dosFP, L):

    rho0 = omArr**2 / np.pi**2 / 3. / consts.c**3

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(omArr, dosFP / rho0, color='indianred', lw=1.)
    ax.set_xlim(np.amin(omArr), np.amax(omArr))
    ax.axhline(1., color = 'black', lw = .5, zorder = -666)
    ax.set_xscale("log")

    ax.set_xlabel(r"$\omega \, \left[\frac{1}{s}\right]$")
    ax.set_ylabel(r"$\rho / \rho_0$")


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/dosFPAsOfOm.png")


def plotDosCompare(omArr, dosFP1, dosFP2, L1, L2):

    omArr = omArr * 1e-12

    lambda1 = 2. * L1
    freq1 = 2. * np.pi * consts.c / (lambda1)
    lambda2 = 2. * L2
    freq2 = 2. * np.pi * consts.c / (lambda2)

    print("res1 = {}THz".format(freq1))
    print("res2 = {}THz".format(freq2))


    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.28, right=0.96)
    ax = plt.subplot(gs[0, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    ax.plot(omArr, dosFP1, color=cmapPink(.5), lw=.7, label = "$d = $" + "{}".format(int(L1 * 1e6)) + r"$\mu \mathrm{m}$")
    ax.plot(omArr, dosFP2, color=cmapBone(.5), lw=.7, label = "$d = $" + "{}".format(int(L2 * 1e6)) + r"$\mu \mathrm{m}$")
    #ax.plot(omArr, (dosFP1 - 0.5) * omArr**0, color=cmapPink(.5), lw=1., label = "$d = $" + "{}".format(int(L1 * 1e6)) + r"$\mu \mathrm{m}$")
    #ax.plot(omArr, (dosFP2 - 0.5) * omArr**0, color=cmapBone(.5), lw=1., label = "$d = $" + "{}".format(int(L2 * 1e6)) + r"$\mu \mathrm{m}$")
    ax.set_xlim(np.amin(omArr), np.amax(omArr))
    ax.axhline(2. / 3., color = 'black', lw = .5, zorder = -666)
    #ax.set_xscale("log")

    ax.set_ylim(0, 2.2)

    ax.set_xticks([0., 100, 200])
    ax.set_xticklabels(["$0$", "$100$", "$200$"], fontsize = 8)
    ax.set_yticks([0., 1., 2.])
    ax.set_yticklabels(["$0$", "$1$", "$2$"], fontsize = 8)

    ax.set_xlabel(r"$\omega \, \left[\mathrm{THz}\right]$")
    #ax.set_ylabel(r"$\left(\rho / \rho_0 - 0.5\right) \times \omega^3$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/dosFPComparePara.png")


def plotDosWithCutoff(omArr, dos):
    omArr = omArr * 1e-12
    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(omArr, dos, color=cmapPink(.5), lw=1.)
    ax.set_xlim(np.amin(omArr), np.amax(omArr))
    ax.axhline(.5, color='black', lw=.5, zorder=-666)
    # ax.set_xscale("log")
    ax.set_xlabel(r"$\omega \, \left[\mathrm{THz}\right]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/dosFPWithCutoff.png")

def plotFieldIntegrals(alphaArr, dosArr):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.24, right=0.96)
    ax = plt.subplot(gs[0, 0])
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(alphaArr * 1e-12, dosArr, color=cmapPink(.5), linestyle = '', marker = 'x', markersize = 2.)
    ax.axhline(0., color = 'red', lw = 0.4)
    #ax.set_ylim(-0.0001, 0.)
    ax.set_xscale('log')

    ax.set_xlabel(r"$\mathrm{cutoff} \; \left[ \mathrm{THz} \right]$")
    ax.set_ylabel(r"$\langle E^2 \rangle \left[ \frac{\mathrm{V}^2}{\mathrm{m}^2} \right] $")

    #ax.text(1e-1, 10, r"$d = 100 \mu \mathrm{m}$")
    #ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/fieldInt.png")
    #plt.savefig("FPPlotsSaved/fieldIntParallelE.png")
    #plt.savefig("FPPlotsSaved/fieldIntPerpE.png")
    #plt.savefig("FPPlotsSaved/fieldIntTEE.png")
    #plt.savefig("FPPlotsSaved/fieldIntTME.png")

def plotFieldWithFixedCutoff(dArr, dosArr, freqArr):

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.22, left=0.18, right=0.85)
    ax = plt.subplot(gs[0, 0])
    axRight = ax.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(dArr, dosArr * 1e-6, color=cmapBone(.6), linestyle = '', marker = 'x', markersize = 2.)
    axRight.plot(dArr, freqArr * 1e-12, color='red', linestyle = '-', lw = 0.5)
    axRight.axhline(241.8, color = 'black', lw = 0.4)
    ax.set_ylim(0., .55 * 1e7 * 1e-6)
    ax.set_xlim(np.amin(dArr), np.amax(dArr))
    ax.set_xscale('log')

    ax.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$")
    ax.set_ylabel(r"$\langle E^2 \rangle \left[ \frac{\mathrm{kV}^2}{\mathrm{m}^2} \right] $")
    axRight.set_ylabel(r"$\omega \left[ \mathrm{THz} \right] $")

    axRight.text(1e-4, 260, r"$\mathrm{cutoff} = 1 \mathrm{eV}$")
    #ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/EFieldAsOfD.png")

def plotEffectiveMass(dArr, delMOverM, freqArr):

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.22, left=0.18, right=0.85)
    ax = plt.subplot(gs[0, 0])
    axRight = ax.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(dArr, delMOverM, color=cmapBone(.6), linestyle = '', marker = 'x', markersize = 2.)
    axRight.plot(dArr, freqArr * 1e-12, color='red', linestyle = '-', lw = 0.5)
    axRight.axhline(241.8, color = 'black', lw = 0.4)
    #ax.set_ylim(0., .55 * 1e7 * 1e-6)
    ax.set_xlim(np.amin(dArr), np.amax(dArr))
    ax.set_xscale('log')

    ax.set_xlabel(r"$d \, \left[ \mathrm{m} \right]$")
    ax.set_ylabel(r"$\frac{\Delta m}{m} $")
    axRight.set_ylabel(r"$\omega \left[ \mathrm{THz} \right] $")

    axRight.text(1e-4, 260, r"$\mathrm{cutoff} = 1 \mathrm{eV}$")
    #ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/effectiveMass.png")


def plotFluctuationsEAsOfD(dArr, dosArr, freqArr, cutoff):

    fig = plt.figure(figsize=(1.7, 1.4), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.8, bottom=0.22, left=0.15, right=0.85)
    ax = plt.subplot(gs[0, 0])
    axRight = ax.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(dArr, dosArr * 1e-6, color=cmapBone(.6), linestyle = '', marker = 'x', markersize = 2.)
    axRight.plot(dArr, freqArr * 1e-12, color='red', linestyle = '-', lw = 0.5)
    axRight.axhline(cutoff * 1e-12, color = 'black', lw = 0.4)
    ax.set_ylim(0., 6)
    ax.set_xlim(1e-6, 1e-4)
    #ax.set_xlim(np.amin(dArr), np.amax(dArr))
    ax.set_xscale('log')

    ax.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$", fontsize = 6)
    ax.yaxis.set_label_coords(0.05, 1.02)
    ax.set_ylabel(r"$\langle E^2 \rangle_{\rm eff} \left[ \frac{\mathrm{MV}^2}{\mathrm{m}^2} \right]$", fontsize = 6, rotation=0, labelpad = 15)
    axRight.yaxis.set_label_coords(1.05, 1.13)
    axRight.set_ylabel(r"$\omega \left[ \mathrm{THz} \right] $", fontsize = 6, rotation=0, labelpad = 0)

    axRight.text(1e-5, cutoff * 1e-12 + 20, r"$\mathrm{cutoff} = 1 \mathrm{eV}$", fontsize = 6)
    #ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    ax.set_yticks([0, 3, 6])
    ax.set_yticklabels(["$0$", "$3$", "$6$"], fontsize = 6)

    axRight.set_yticks([0, 300, 600])
    axRight.set_yticklabels(["$0$", "$300$", "$600$"], fontsize = 6)

    ax.set_xticks([1e-6, 1e-5, 1e-4])
    ax.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$"], fontsize = 6)

    arrowprops = {
        'arrowstyle': '<-',  # Arrow style
        'mutation_scale': 8,  # Custom head size
        'color': 'red',  # Custom arrow color
        'linewidth': .7
    }
    ax.annotate("", xy=(3. * 1e-5, .5), xytext=(1e-4, 1.5), arrowprops=arrowprops)
    arrowprops = {
        'arrowstyle': '<-',  # Arrow style
        'mutation_scale': 8,  # Custom head size
        'color': cmapBone(.6),  # Custom arrow color
        'linewidth': .7
    }
    ax.annotate("", xy=(3. * 1e-6, .5), xytext=(1e-6, 1.5), arrowprops=arrowprops)
    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        axRight.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/FluctuationsE.png")


def plotFluctuationsAAsOfD(dArr, dosArr, freqArr, cutoff):

    fig = plt.figure(figsize=(1.7, 1.4), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.8, bottom=0.22, left=0.2, right=0.85)
    ax = plt.subplot(gs[0, 0])
    axRight = ax.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(dArr, dosArr, color=cmapBone(.6), linestyle = '', marker = 'x', markersize = 2.)
    axRight.plot(dArr, freqArr * 1e-12, color='red', linestyle = '-', lw = 0.5)
    axRight.axhline(cutoff * 1e-12, color = 'black', lw = 0.4)
    #ax.set_ylim(0., 1e4)
    ax.set_xlim(1e-6, 1e-4)
    axRight.set_xlim(1e-6, 1e-4)
    #ax.set_xlim(np.amin(dArr), np.amax(dArr))
    ax.set_xscale('log')

    ax.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$", fontsize = 6)
    ax.yaxis.set_label_coords(0.02, 1.02)
    ax.set_ylabel(r"$\langle A^2 \rangle_{\rm eff} \left[ \frac{\mathrm{kV}^2}{\mathrm{THz}^2 \, \mathrm{m}^2} \right]$", fontsize = 6, rotation=0, labelpad = 15)
    axRight.yaxis.set_label_coords(1.1, 1.13)
    axRight.set_ylabel(r"$\omega \left[ \mathrm{THz} \right] $", fontsize = 6, rotation=0, labelpad = 0)

    axRight.text(1e-5, cutoff * 1e-12 + 20, r"$\mathrm{cutoff} = 1 \mathrm{eV}$", fontsize = 6)

    ax.set_yticks([0, -200, -400])
    ax.set_yticklabels(["$0$", "$-200$", "$- 400$"], fontsize = 6)

    axRight.set_yticks([0, 300, 600])
    axRight.set_yticklabels(["$0$", "$300$", "$600$"], fontsize = 6)

    ax.set_xticks([1e-6, 1e-5, 1e-4])
    ax.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$"], fontsize = 6)

    arrowprops = {
        'arrowstyle': '<-',  # Arrow style
        'mutation_scale': 8,  # Custom head size
        'color': 'red',  # Custom arrow color
        'linewidth': .7
    }
    ax.annotate("", xy=(3. * 1e-5, -400), xytext=(1e-4, -350), arrowprops=arrowprops)
    arrowprops = {
        'arrowstyle': '<-',  # Arrow style
        'mutation_scale': 8,  # Custom head size
        'color': cmapBone(.6),  # Custom arrow color
        'linewidth': .7
    }
    ax.annotate("", xy=(2. * 1e-6, -300), xytext=(1e-6, -200), arrowprops=arrowprops)

    #ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)
        axRight.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/FluctuationsA.png")



def plotFluctuationsAandE(dArr, flucE, flucA, freqArr, cutoff):

    fig = plt.figure(figsize=(3.5, 2.), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1],
                           wspace=0.35, hspace=0., top=0.95, bottom=0.18, left=0.2, right=0.85)
    axE = plt.subplot(gs[0, 0])
    axA = plt.subplot(gs[1, 0])
    axRightE = axE.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    axE.plot(dArr, flucE * 1e-6, color=cmapBone(.55), linestyle = '', marker = 'x', markersize = 2.)
    axA.plot(dArr, flucA, color=cmapPink(.55), linestyle = '', marker = 'x', markersize = 2.)
    axRightE.plot(dArr, freqArr * 1e-12, color='red', linestyle = '-', lw = 0.5)
    axRightE.axhline(cutoff * 1e-12, color = 'black', lw = 0.4)
    axE.set_ylim(-50., 1000)
    axA.set_ylim(-200, 20)
    #axRightE.set_xlim(1e-6, 1e-4)
    axE.set_xlim(np.amin(dArr), np.amax(dArr))
    axA.set_xlim(np.amin(dArr), np.amax(dArr))
    axE.set_xscale('log')
    axA.set_xscale('log')

    axA.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$", fontsize = 8)
    axE.set_ylabel(r"$\langle E^2 \rangle_{\rm eff} \left[ \frac{\mathrm{kV}^2}{\mathrm{m}^2} \right]$", fontsize = 8, labelpad = 5)
    axA.set_ylabel(r"$\langle A^2 \rangle_{\rm eff} \left[ \frac{\mathrm{V}^2}{\mathrm{THz}^2 \, \mathrm{m}^2} \right]$", fontsize = 8, labelpad = 2)
    axRightE.set_ylabel(r"$\omega_0 \left[ \mathrm{THz} \right] $", fontsize = 8, labelpad = 4)

    axRightE.text(1e-5, cutoff * 1e-12 - 300, r"$\mathrm{cutoff} = 10 \mathrm{eV}$", fontsize = 8)


    axA.set_yticks([0, 3, 6])
    axA.set_yticklabels(["$0$", "$3$", "$6$"], fontsize = 8)

    axA.set_yticks([0, -100])
    axA.set_yticklabels(["$0$", "$-100$"], fontsize = 8)

    axRightE.set_yticks([0, 1000, 2000])
    axRightE.set_yticklabels(["$0$", "$1000$", "$2000$"], fontsize = 8)

    axA.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axA.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize = 8)


    axE.set_xticks([])


    arrowprops = {
        'arrowstyle': '<-',  # Arrow style
        'mutation_scale': 8,  # Custom head size
        'color': 'red',  # Custom arrow color
        'linewidth': .7
    }
    axE.annotate("", xy=(3. * 1e-5, -400), xytext=(1e-4, -350), arrowprops=arrowprops)
    arrowprops = {
        'arrowstyle': '<-',  # Arrow style
        'mutation_scale': 8,  # Custom head size
        'color': cmapBone(.6),  # Custom arrow color
        'linewidth': .7
    }
    axE.annotate("", xy=(2. * 1e-6, -300), xytext=(1e-6, -200), arrowprops=arrowprops)

    #ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    #legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)
        axA.spines[axis].set_linewidth(.5)
        axRightE.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/FluctuationsEandA.png")


def plotFluctuationsAandENaturalUnits(dArr, flucE, flucA, freqArr, cutoff):
    fig = plt.figure(figsize=(7.04 / 3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1, wspace=0.35, hspace=0., top=0.9, bottom=0.18, left=0.2, right=0.85)
    axE = plt.subplot(gs[0, 0])
    axA = axE.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    unitFacE = consts.epsilon_0 * dArr**3 / (consts.hbar * freqArr)
    unitFacA = consts.epsilon_0 * freqArr * dArr**3 / consts.hbar

    axE.plot(dArr, flucE * unitFacE, color=cmapBone(.55), linestyle='', marker='x', markersize=2.)
    #axA.plot(dArr, flucA * unitFacA * 1e-24, color=cmapPink(.55), linestyle='', marker='x', markersize=2.)
    axE.set_ylim(0.06, .16)
    #axA.set_ylim(-0.02, 0.02)
    # axRightE.set_xlim(1e-6, 1e-4)
    axE.set_xlim(np.amin(dArr), np.amax(dArr))
    axA.set_xlim(np.amin(dArr), np.amax(dArr))
    axE.set_xscale('log')
    #axE.set_yscale('log')
    axA.set_xscale('log')

    axA.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$", fontsize=8)
    axE.set_ylabel(r"$\langle E^2 \rangle_{\rm eff} \left[ \frac{\hbar \omega_0}{\varepsilon_0 d^3} \right]$", fontsize=8,
                   labelpad=8)
    axA.set_ylabel(
        r"$\langle A^2 \rangle_{\rm eff} \left[ \frac{\hbar}{\varepsilon_0 \omega_0 d^3} \right]$",
        fontsize=8, labelpad=0)

    #axA.set_yticks([0, 3, 6])
    #axA.set_yticklabels(["$0$", "$3$", "$6$"], fontsize=8)

    #axA.set_yticks([0, -1000])
    #axA.set_yticklabels(["$0$", "$-1000$"], fontsize=8)

    #axE.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    #axE.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize=8)

    #arrowprops = {
    #    'arrowstyle': '<-',  # Arrow style
    #    'mutation_scale': 8,  # Custom head size
    #    'color': 'red',  # Custom arrow color
    #    'linewidth': .7
    #}
    #axE.annotate("", xy=(3. * 1e-5, -400), xytext=(1e-4, -350), arrowprops=arrowprops)
    #arrowprops = {
    #    'arrowstyle': '<-',  # Arrow style
    #    'mutation_scale': 8,  # Custom head size
    #    'color': cmapBone(.6),  # Custom arrow color
    #    'linewidth': .7
    #}
    #axE.annotate("", xy=(2. * 1e-6, -300), xytext=(1e-6, -200), arrowprops=arrowprops)

    # ax.text(1e-1, 7.5, r"$\omega_0 = 9.41 \mathrm{THz}$")

    # legend = ax.legend(fontsize=fontsize - 2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black',
    #                   ncol=1)
    # legend.get_frame().set_alpha(0.)
    # legend.get_frame().set_boxstyle('Square', pad=0.1)
    # legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)
        axA.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/FluctuationsEandANatUnits.png")


def plotEffectiveMassesComparison():

    filenameFP =  "delMassFP"
    cutoff, dArr, massFPArr = handleIntegralData.retrieveMassData(filenameFP)

    filenameSPhP = "massesForPlotting"
    cutoff, zArr, massSPhPArr = handleIntegralData.readMassesSPhP(filenameSPhP)

    fig = plt.figure(figsize=(2., 2.), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                           wspace=0.35, hspace=0.6, top=0.92, bottom=0.2, left=0.22, right=0.92)
    axFP = plt.subplot(gs[0, 0])
    axSPhP = plt.subplot(gs[1, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axFP.plot(dArr, massFPArr * 1e7, color = cmapPink(.5), lw = 0.8)

    axSPhP.plot(zArr, massSPhPArr, color = cmapBone(.5), lw = 0.8)

    axFP.set_xscale("log")
    axSPhP.set_xscale("log")

    axFP.set_xlim(np.amin(dArr), np.amax(dArr))
    axFP.set_ylim(-8., 0.)
    axSPhP.set_xlim(1e-8, 1e-3)
    axSPhP.set_ylim(0., 1.)

    axFP.set_xlabel("$d [\mathrm{m}]$", fontsize = 8)
    axSPhP.set_xlabel("$z [\mathrm{m}]$", fontsize = 8)

    axFP.set_ylabel(r"$\frac{\Delta m}{m}$", labelpad = 0., fontsize = 8)
    axSPhP.set_ylabel(r"$\frac{\Delta m}{m}$", labelpad = 6., fontsize = 8)

    axFP.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axFP.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize = 8)

    axFP.set_yticks([0., -5.])
    axFP.set_yticklabels(["$0$", "$-5$"], fontsize = 8)

    axSPhP.set_xticks([1e-8, 1e-6, 1e-4])
    axSPhP.set_xticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$"], fontsize = 8)

    axSPhP.set_yticks([0., 1.])
    axSPhP.set_yticklabels(["$0$", "$1$"], fontsize = 8)

    axFP.text(1.1 * 1e-6, -1.4, r"$\times 10^{-7}$", fontsize = 8)
    axFP.text(1. * 1e-5, -5., r"$\mathrm{Fabry-Perot}$", fontsize = 8)
    axSPhP.text(6. * 1e-7, .5, r"$\mathrm{Surface}$", fontsize = 8)



    for axis in ['top', 'bottom', 'left', 'right']:
        axFP.spines[axis].set_linewidth(.5)
        axSPhP.spines[axis].set_linewidth(.5)


    plt.savefig("FPPlotsSaved/EffectiveMassesCompared.png")


def plotEffectiveHoppingComparison():

    filenameFP =  "delHoppingFP"
    cutoff, dArr, massFPArr = handleIntegralData.retrieveMassData(filenameFP)

    filenameSPhP = "hoppingForPlotting"
    cutoff, zArr, massSPhPArr = handleIntegralData.readMassesSPhP(filenameSPhP)

    fig = plt.figure(figsize=(3., 2.2), dpi=800)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1],
                           wspace=0.35, hspace=0.6, top=0.95, bottom=0.18, left=0.22, right=0.92)
    axFP = plt.subplot(gs[0, 0])
    axSPhP = plt.subplot(gs[1, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axFP.plot(dArr, -massFPArr * 1e12, color = cmapPink(.5), lw = 0.8)

    axSPhP.plot(zArr, -massSPhPArr * 1e2, color = cmapBone(.5), lw = 0.8)

    axFP.set_xscale("log")
    axSPhP.set_xscale("log")

    axFP.set_xlim(np.amin(dArr), np.amax(dArr))
    axFP.set_ylim(0., 10.)
    axSPhP.set_xlim(1e-9, 1e-3)
    axSPhP.set_ylim(-0.4, 0.)

    axFP.set_xlabel("$d [\mathrm{m}]$", fontsize = 8)
    axSPhP.set_xlabel("$z [\mathrm{m}]$", fontsize = 8)

    axFP.set_ylabel(r"$\frac{\Delta t}{t}[\%]$", labelpad = 2., fontsize = 8)
    axSPhP.set_ylabel(r"$\frac{\Delta t}{t}[\%]$", labelpad = 2., fontsize = 8)

    axFP.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    axFP.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize = 8)

    axFP.set_yticks([0., 10.])
    axFP.set_yticklabels(["$0$", "$10$"], fontsize = 8)

    axSPhP.set_xticks([1e-8, 1e-6, 1e-4])
    axSPhP.set_xticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$"], fontsize = 8)

    axSPhP.set_yticks([-0.25, 0.])
    axSPhP.set_yticklabels(["$-0.25$", "$0$"], fontsize = 8)

    axFP.text(1.3 * 1e-6, 8., r"$\times 10^{-10}$", fontsize = 8)
    axFP.text(1. * 1e-5, 5., r"$\mathrm{Fabry-Perot}$", fontsize = 8)
    axSPhP.text(10. * 1e-8, -.2, r"$\mathrm{Surface}$", fontsize = 8)



    for axis in ['top', 'bottom', 'left', 'right']:
        axFP.spines[axis].set_linewidth(.5)
        axSPhP.spines[axis].set_linewidth(.5)


    plt.savefig("FPPlotsSaved/EffectiveHoppingCompared.png")


