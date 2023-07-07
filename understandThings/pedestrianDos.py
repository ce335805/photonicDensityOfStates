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

def dispersionK(kx, ky, kz, eps):
    return consts.c / np.sqrt(eps) * np.sqrt(kx**2 + ky**2 + kz**2)

def dosPedestrian(kxArr, kyArr, kzArr, zArr, omega, deltaOmega, eps, L):
    dos = np.zeros(zArr.shape)
    for kx in kxArr:
        for ky in kyArr:
            for kz in kzArr:
                omegaTry = dispersionK(kx, ky, kz, eps)
                if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                #if (omegaTry > omega):
                        continue
                dos += 2. * np.sin(kz * zArr)**2 / L**3
                #dos += 1 / L ** 3
    return dos

def dosAnalytical(omega, zVal, eps):
    keff = 2. * np.sqrt(eps) * omega / consts.c
    return np.sqrt(eps)**3 / 2 * omega**2 / (np.pi**2 * consts.c**3) * (1 - np.sin(keff * zVal) / (keff * zVal))

def dosAnalyticalInt(zArr, omega, deltaOmega, eps):
    dosInt = np.zeros(zArr.shape)
    for zInd, zVal in enumerate(zArr):
        dosInt[zInd] = integrate.quad(dosAnalytical, omega, omega + deltaOmega, args=(zVal, eps))[0]
    return dosInt

def plotDos(zArr, dos, L, omega, deltaOmega, eps):

    dosAna = dosAnalyticalInt(zArr, omega, deltaOmega, eps)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    freeRes = 1. / (np.pi**2 * consts.c**3) * 1. / 3. * ((omega + deltaOmega)**3 - omega**3)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.plot(zArr, dosAna, color='teal', lw=1., label = "DOS from Box", linestyle = '--')
    #ax.axhline(freeRes, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps), lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**3, lw = 0.5, color = 'gray', zorder = -666)

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    #ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    #ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$z[m]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.savefig("./savedPlots/dosPedestrian.png")

def pedestrianMain():

    L = 0.2
    eps = 1.
    omega = 1e11
    deltaOmega = omega / 10.

    kBound = np.sqrt(eps) * (omega + deltaOmega) / consts.c
    nBound = math.ceil(kBound * L / (2. * np.pi))
    kzInds = np.arange(1., 2. * nBound)
    kzArr = kzInds * np.pi / L
    kxInds = np.arange(- nBound, nBound)
    kxArr = kxInds * 2. * np.pi / L
    kyArr = kxArr
    zArr = np.linspace(0., .1, 500)
    print(kxArr.shape)

    dos = dosPedestrian(kxArr, kyArr, kzArr, zArr, omega, deltaOmega, eps, L)
    plotDos(zArr, dos, L, omega, deltaOmega, eps)