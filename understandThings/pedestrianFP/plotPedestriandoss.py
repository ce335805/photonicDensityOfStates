import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad
import math

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


def plotPedestrianDoss(zArr, dos, omega, deltaOmega, eps):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    rho0 = (omega + deltaOmega / 2)**2 / (2. * np.pi**2 * consts.c ** 3)

    ax.plot(zArr, dos / deltaOmega / rho0, color='peru', lw=1., label = "DOS from Box")
    #ax.plot(zArr, dosInt / deltaOmega / rho0, color='coral', lw=1., label = "DOS from Box")
    #ax.plot(zArr, dosAna / deltaOmega / rho0, color='teal', lw=1., label = "DOS from Box", linestyle = '--')
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps), lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**3, lw = 0.5, color = 'gray', zorder = -666)

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    #ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    #ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$z[m]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.savefig("./savePedestrianFPPlots/dosPed.png")

