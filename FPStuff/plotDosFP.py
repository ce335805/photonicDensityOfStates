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

    ax.plot(omArr, dosFP1, color=cmapPink(.5), lw=1., label = "$d = $" + "{}".format(int(L1 * 1e6)) + r"$\mu \mathrm{m}$")
    ax.plot(omArr, dosFP2, color=cmapBone(.5), lw=1., label = "$d = $" + "{}".format(int(L2 * 1e6)) + r"$\mu \mathrm{m}$")
    #ax.plot(omArr, (dosFP1 - 0.5) * omArr**0, color=cmapPink(.5), lw=1., label = "$d = $" + "{}".format(int(L1 * 1e6)) + r"$\mu \mathrm{m}$")
    #ax.plot(omArr, (dosFP2 - 0.5) * omArr**0, color=cmapBone(.5), lw=1., label = "$d = $" + "{}".format(int(L2 * 1e6)) + r"$\mu \mathrm{m}$")
    ax.set_xlim(np.amin(omArr), np.amax(omArr))
    ax.axhline(.5, color = 'black', lw = .5, zorder = -666)
    #ax.set_xscale("log")

    ax.set_ylim(0, 1.1)

    ax.set_xlabel(r"$\omega \, \left[\mathrm{THz}\right]$")
    #ax.set_ylabel(r"$\left(\rho / \rho_0 - 0.5\right) \times \omega^3$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(fontsize=fontsize-2, loc='upper right', bbox_to_anchor=(.97, 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/dosFPCompareTE.png")


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


def plotHoppingWithFixedCutoff(dArr, dosArr, freqArr):

    fig = plt.figure(figsize=(3.4, 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.22, left=0.18, right=0.85)
    ax = plt.subplot(gs[0, 0])
    axRight = ax.twinx()
    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    ax.plot(dArr, - dosArr, color=cmapBone(.6), linestyle = '', marker = 'x', markersize = 2.)
    axRight.plot(dArr, freqArr * 1e-12, color='red', linestyle = '-', lw = 0.5)
    axRight.axhline(241.8, color = 'black', lw = 0.4)
    #ax.set_ylim(- 0.01 * 1e-18, 0.)
    ax.set_xlim(np.amin(dArr), np.amax(dArr))
    ax.set_xscale('log')

    ax.set_xlabel(r"$d \; \left[ \mathrm{m} \right]$")
    ax.set_ylabel(r"$\frac{\Delta t}{t_0}$")
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
        axRight.spines[axis].set_linewidth(.5)

    plt.savefig("FPPlotsSaved/HoppingOfD.png")

