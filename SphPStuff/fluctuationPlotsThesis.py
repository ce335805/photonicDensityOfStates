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



def plotFluctuationsENaturalUnits(dosNoSurfE, dosTotE, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(6.5 / 3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.95, bottom=0.22, left=0.3, right=0.95)
    axE = plt.subplot(gs[0, 0])

    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf
    natUnitFac = consts.epsilon_0 * lambda0**3 / (consts.hbar * wInf)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axE.plot(zArr / lambda0, np.abs(dosTotE) * natUnitFac, color=cmapBone(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Total}$")
    axE.plot(zArr / lambda0, np.abs(dosNoSurfE) * natUnitFac, color=cmapPink(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Bulk}$")

    axE.set_xscale("log")
    axE.set_yscale("log")
    axE.set_xlim(1e-3, 1e1)
    axE.set_ylim(1e-4, 1e8)
    #axE.set_ylim(1e10, 1e20)

    axE.plot(zArr / lambda0, np.abs(dosTotE[-1]) * zArr[-1]**3 / zArr**3 * natUnitFac, color = 'red', lw = 0.3)
    axE.plot(zArr / lambda0, np.abs(dosNoSurfE[-50]) * zArr[-50]**2 / zArr**2 * natUnitFac, color = 'red', lw = 0.3)

    axE.set_xlabel(r"$z[\lambda_0]$", fontsize = 8)
    axE.set_ylabel(r"$ \langle \widehat{E \,}^{\! 2} \rangle_{\rm eff}  \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \, \omega_{\mathrm{S}}}{\varepsilon_0 \lambda_0^3}\right] $", fontsize = 8, labelpad = 1)

    #axE.set_xticks([])
    #axE.set_yticks([1e-10, 1e-5, 1.])
    #axE.set_yticklabels(["$10^{-10}$", "$10^{-5}$", "$1$"], fontsize = 8)

    #axE.text(1e-1, 1e1, r"$\sim z^{-3}$", fontsize = 7, color = "red")
    #axE.text(1.5 * 1e-2, .3 * 1e-1, r"$\sim z^{-2}$", fontsize = 7, color = "red")

    legend = axE.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    axE.text(-0.45, 0.98, r"$\mathrm{(b)}$", fontsize = 8, transform = axE.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/FluctuationsENatUnits.png")


def plotFluctuationsANaturalUnits(dosNoSurfA, dosTotA, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(6.5 / 3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.95, bottom=0.22, left=0.38, right=0.95)
    axA = plt.subplot(gs[0, 0])

    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    lambda0 = consts.c / (2. * np.pi * wInf)
    natUnitFac = consts.epsilon_0 * lambda0**3 * wInf / (consts.hbar)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axA.plot(zArr / lambda0, dosTotA * natUnitFac, color=cmapBone(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Total}$")
    axA.plot(zArr / lambda0, dosNoSurfA * natUnitFac, color=cmapPink(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Bulk}$")

    axA.set_xscale("log")
    #axA.set_yscale("log")
    axA.set_xlim(1e-2, 1e2)
    axA.set_ylim(-0.15 * 1e-3, .15 * 1e-3)

    axA.plot(zArr / lambda0, np.abs(dosTotA[110]) * zArr[110]**3 / zArr**3 * natUnitFac, color = 'red', lw = 0.3)

    ydat = np.abs(dosNoSurfA)
    fitInd = 160
    logZArr = np.log(zArr)
    slope = (ydat[fitInd] - ydat[fitInd - 1]) / (logZArr[fitInd] - logZArr[fitInd - 1])
    offset = ydat[fitInd] - slope * logZArr[fitInd]
    #axA.plot(np.exp(logZArr), - slope * logZArr + 0. * offset, color = 'red', lw = 0.5)
    axA.plot(zArr / lambda0, -(slope * logZArr + offset) * natUnitFac, color = 'red', lw = 0.3)

    axA.set_xlabel(r"$z[\lambda_0]$", fontsize = 8)
    axA.set_ylabel(r"$ \langle \widehat{A \,}^{\! 2} \rangle_{\rm eff}  \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt}}{\varepsilon_0 \omega_{\mathrm{S}} \lambda_0^3}\right] $", fontsize = 8, labelpad = 1)

    #axE.set_xticks([])
    #axE.set_yticks([1e-10, 1e-5, 1.])
    #axE.set_yticklabels(["$10^{-10}$", "$10^{-5}$", "$1$"], fontsize = 8)

    axA.text(.5 * 1e0, 1.2 * 1e-4, r"$\sim z^{-3}$", fontsize = 7, color = "red")
    axA.text(1e-1, 1.2 * -1e-4, r"$\sim \mathrm{log}(z) + \mathrm{const}$", fontsize = 7, color = "red")

    legend = axA.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0., .8), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    axA.text(-0.65, 0.98, r"$\mathrm{(c)}$", fontsize = 8, transform = axA.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axA.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/FluctuationsANatUnits.png")

def plotFluctuationsEExpUnits(dosNoSurfE, dosTotE, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(2., 1.5), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.95, bottom=0.25, left=0.25, right=0.9)
    axE = plt.subplot(gs[0, 0])

    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf
    natUnitFac = consts.epsilon_0 * lambda0**3 / (consts.hbar * wInf)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')
    axE.plot(zArr, np.sqrt(np.abs(dosTotE)) * 1e-6, color=cmapBone(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Total}$")
    #axE.plot(zArr, np.abs(dosNoSurfE), color=cmapPink(.5), linestyle='', marker='x', markersize=2., label = r"$\mathrm{Bulk}$")

    axE.set_xscale("log")
    #axE.set_yscale("log")
    axE.set_xlim(1e-9, 1e-6)
    axE.set_ylim(0., 50)

    axE.set_xlabel(r"$z \, [\mathrm{m}]$", fontsize = 8)
    axE.set_ylabel(r"$ \sqrt{\langle \widehat{E \,}^{\! 2} \rangle_{\rm eff}}  \left[\frac{\mathrm{MV}}{\mathrm{m}}\right] $", fontsize = 8, labelpad = 4)

    axE.set_xticks([1e-9, 1e-8, 1e-7, 1e-6])
    axE.set_xticklabels(["$10^{-9}$", "$10^{-8}$", "$10^{-7}$", "$10^{-6}$"], fontsize = 8)
    axE.set_yticks([0, 20, 40])
    axE.set_yticklabels(["$0$", "$20$", "$40$"], fontsize = 8)

    axE.text(.5, .5, r"$\mathrm{Surface}$", fontsize = 8, transform = axE.transAxes, ha='center', va='center')
    axE.text(-0.36, 0.98, r"$\mathrm{(b)}$", fontsize = 8, transform = axE.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/FluctuationsEExpUnits.png")


def plotFluctuationsECutoffConv(dosNoSurfEArr, dosTotEArr, cutoffArr, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.95, bottom=0.22, left=0.24, right=0.95)
    axE = plt.subplot(gs[0, 0])

    evCutoff = 1519.3 * 1e12

    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf
    natUnitFac = consts.epsilon_0 * lambda0**3 / (consts.hbar * wInf)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    for cutoffInd, cutoff in enumerate(cutoffArr):
        color1 = cmapBone((cutoffInd + 1.) / (len(cutoffArr) + 2.))
        color2 = cmapPink((cutoffInd + 1.) / (len(cutoffArr) + 2.))
        axE.plot(zArr / lambda0, np.abs(dosTotEArr[cutoffInd, :]) * natUnitFac, color=color1, linestyle='', marker='x', markersize=2.)
        axE.plot(zArr / lambda0, np.abs(dosNoSurfEArr[cutoffInd, :]) * natUnitFac, color=color2, linestyle='', marker='x', markersize=2.
                 , label = r"$\Lambda = $" + "{:.3f}".format(cutoff / evCutoff) + "$\mathrm{eV}$")

    axE.set_xscale("log")
    axE.set_yscale("log")
    axE.set_xlim(1e-3, 1e1)
    axE.set_ylim(1e-4, 1e8)
    #axE.set_ylim(1e10, 1e20)

    axE.set_xlabel(r"$z[\lambda_0]$", fontsize = 8)
    axE.set_ylabel(r"$ \langle \widehat{E \,}^{\! 2} \rangle_{\rm eff}  \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \, \omega_{\mathrm{S}}}{\varepsilon_0 \lambda_0^3}\right] $", fontsize = 8, labelpad = 1)

    #axE.set_xticks([])
    #axE.set_yticks([1e-10, 1e-5, 1.])
    #axE.set_yticklabels(["$10^{-10}$", "$10^{-5}$", "$1$"], fontsize = 8)

    #axE.text(1e-1, 1e1, r"$\sim z^{-3}$", fontsize = 7, color = "red")
    #axE.text(1.5 * 1e-2, .3 * 1e-1, r"$\sim z^{-2}$", fontsize = 7, color = "red")

    legend = axE.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.05), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    axE.text(-0.45, 0.98, r"$\mathrm{(b)}$", fontsize = 8, transform = axE.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axE.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/FluctuationsECutoffConv.png")


def plotFluctuationsACutoffConv(dosNoSurfAArr, dosTotAArr, cutoffArr, zArr, L, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3., 1.8), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.95, bottom=0.22, left=0.24, right=0.95)
    axA = plt.subplot(gs[0, 0])

    evCutoff = 1519.3 * 1e12

    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    lambda0 = consts.c / (2. * np.pi * wInf)
    natUnitFac = consts.epsilon_0 * lambda0**3 * wInf / (consts.hbar)

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    for cutoffInd, cutoff in enumerate(cutoffArr):
        color1 = cmapBone((cutoffInd + 1.) / (len(cutoffArr) + 2.))
        color2 = cmapPink((cutoffInd + 1.) / (len(cutoffArr) + 2.))
        axA.plot(zArr / lambda0, dosTotAArr[cutoffInd, :] * natUnitFac, color=color1, linestyle='', marker='x', markersize=2.)
        axA.plot(zArr / lambda0, dosNoSurfAArr[cutoffInd, :] * natUnitFac, color=color2, linestyle='', marker='x', markersize=2.
                 , label = r"$\Lambda = $" + "{:.3f}".format(cutoff / evCutoff) + "$\mathrm{eV}$")

    axA.set_xscale("log")
    #axA.set_yscale("log")
    axA.set_xlim(1e-2, 1e2)
    axA.set_ylim(-0.15 * 1e-3, 0.15 * 1e-3)

    axA.set_xlabel(r"$z[\lambda_0]$", fontsize = 8)
    axA.set_ylabel(r"$ \langle \widehat{A \,}^{\! 2} \rangle_{\rm eff}  \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt}}{\varepsilon_0 \omega_{\mathrm{S}} \lambda_0^3}\right] $", fontsize = 8, labelpad = 1)

    #axE.set_xticks([])
    #axE.set_yticks([1e-10, 1e-5, 1.])
    #axE.set_yticklabels(["$10^{-10}$", "$10^{-5}$", "$1$"], fontsize = 8)

    #axE.text(1e-1, 1e1, r"$\sim z^{-3}$", fontsize = 7, color = "red")
    #axE.text(1.5 * 1e-2, .3 * 1e-1, r"$\sim z^{-2}$", fontsize = 7, color = "red")

    #legend = axA.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.05), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    axA.text(-0.45, 0.98, r"$\mathrm{(b)}$", fontsize = 8, transform = axA.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axA.spines[axis].set_linewidth(.5)

    plt.savefig("./ThesisPlots/FluctuationsACutoffConv.png")

