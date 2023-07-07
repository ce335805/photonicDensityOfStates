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
import complexIntegral
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.constants as consts

import SphPStuff.findAllowedKsSPhP as findAllowedKsSPhP
import SphPStuff.epsilonFunctions as epsFunc

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

from SphPStuff.dosFuncs import dosTEModes as dosTE
from SphPStuff.dosFuncs import dosTEEvaModes as dosTEEva
from SphPStuff.dosFuncs import dosTEResModes as dosTERes
from SphPStuff.dosFuncs import dosTMModes as dosTM
from SphPStuff.dosFuncs import dosTMEvaModes as dosTMEva
from SphPStuff.dosFuncs import dosTMResModes as dosTMRes
from SphPStuff.dosFuncs import dosTMSurfModes as dosTMSurf


def getDosTM(zVal, L, omega, wLO, wTO, epsInf):
    if(omega < wTO or omega > wLO ):
        return dosTM.calcDosTM(np.array([zVal]), L, omega, wLO, wTO, epsInf)
    else:
        return 0
def getDosTE(zVal, L, omega, wLO, wTO, epsInf):
    if(omega < wTO or omega > wLO ):
        return dosTE.calcDosTE(np.array([zVal]), L, omega, wLO, wTO, epsInf)
    else:
        return 0


def createPlotAsOfOmega():

    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = .5
    zVal = L * 1e-1

    omegaArr = np.linspace(0.01, 0.5, 156) * 1e13
    #omegaArr = np.array([0.32 * 1e13])

    dos = np.zeros(omegaArr.shape)
    for omegaInd, omegaVal in enumerate(omegaArr):
        print("omega = {}THz".format(omegaVal * 1e-12))
        dos[omegaInd] = getDosTE(zVal, L, omegaVal, wLO, wTO, epsInf)

    plotDosAsOfFreq(dos, zVal, L, omegaArr, wLO, wTO, epsInf)
    print(dos)

def plotDosAsOfFreq(dos, zVal, L, omegaArr, wLO, wTO, epsInf):

    omegaArr = omegaArr * 1e-12 #convert to THz

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(omegaArr, dos, color='peru', lw=1., label = "DOS from Box")

    ax.axvline(wLO * 1e-12, lw = 0.5, color = 'gray')
    ax.axvline(wTO * 1e-12, lw = 0.5, color = 'gray')
    print("eps = 1 at {}THz".format(np.sqrt((epsInf * wLO**2 - wTO**2) / (epsInf - 1)) * 1e-12))
    ax.axvline(np.sqrt((epsInf * wLO**2 - wTO**2) / (epsInf - 1)) * 1e-12, lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(omegaArr), np.amax(omegaArr))

    #ax.set_xticks([np.amin(omegaArr), 0, np.amax(omegaArr)])
    #ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$\omega[\mathrm{THz}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosAsOfFreqL05.png")