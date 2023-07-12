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

import understandThings.allowedKsKPara as allowedKsK
import understandThings.allowedKsOmega as allowedKsOm

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


def dispersionK(kx, ky, kz):
    return consts.c * np.sqrt(kx**2 + ky**2 + kz**2)

def dispersionKKPara(kPara, kz):
    return consts.c * np.sqrt(kPara**2 + kz**2)


def dosPedestrian(kxArr, kyArr, zArr, omega, deltaOmega, eps, L):
    dos = np.zeros(zArr.shape)
    print("Computing dos via sum")
    for kx in kxArr:
        for ky in kyArr:
            kzArr = allowedKsK.findKs(L, omega + deltaOmega, kx, ky, eps, "TE")
            for kz in kzArr:
                omegaTry = dispersionK(kx, ky, kz)
                if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                        continue
                kD = np.sqrt((eps - 1) * (kx**2 + ky**2) + eps * kz ** 2)
                NSqr = L**3 / 2 * (eps * np.sin(kz * L / 2.) ** 2 * (1 - np.sin(kD * L) / (kD * L)) + np.sin(kD * L / 2.) ** 2  * (1 - np.sin(kz * L) / (kz * L)))
                dos += 1 / NSqr * np.sin(kz * (L / 2. - zArr))**2 * np.sin(kD * L / 2.) ** 2
    return dos

def dosPedestrianInt(zArr, omega, deltaOmega, eps, L):
    print("Computing dos via kPara Integral")
    kParaArr = np.linspace(0, (omega + deltaOmega) / consts.c, 200)
    dos = np.zeros((len(zArr), len(kParaArr)))
    for kParaInd, kParaVal in enumerate(kParaArr):
        kzArr = allowedKsK.findKsKPara(L, omega + deltaOmega, kParaVal**2, eps, "TE")
        for kz in kzArr:
            omegaTry = dispersionKKPara(kParaVal, kz)
            if(omegaTry <= omega or omegaTry >= omega + deltaOmega):
                    continue
            kD = np.sqrt((eps - 1) * kParaVal**2 + eps * kz ** 2)
            NSqr = L**3 / 2 * (eps * np.sin(kz * L / 2.) ** 2 * (1 - np.sin(kD * L) / (kD * L)) + np.sin(kD * L / 2.) ** 2  * (1 - np.sin(kz * L) / (kz * L)))
            dos[:, kParaInd] += 1 / NSqr * np.sin(kz * (L / 2. - zArr))**2 * np.sin(kD * L / 2.) ** 2
    dosSummed = np.trapz(L**2 / (2. * np.pi) * kParaArr[None, :] * dos[:, :], kParaArr, axis = 1)
    return dosSummed

def dosPedestrianOmInt(zArr, omega, deltaOmega, eps, L):
    print("Computing dos via omega Integral")
    omArr = np.linspace(omega, omega + deltaOmega, 100)
    dos = np.zeros((len(zArr), len(omArr)))
    for omInd, omVal in enumerate(omArr):
        kzArr = allowedKsOm.findKs(L, omVal, eps, "TE")
        for kz in kzArr:
            #if(omVal < omega or omVal > omega + deltaOmega):
            #        continue
            kD = np.sqrt((eps - 1) * omVal**2 / consts.c**2 + kz ** 2)
            NSqr = L**3 / 2 * (eps * np.sin(kz * L / 2.) ** 2 * (1 - np.sin(kD * L) / (kD * L)) + np.sin(kD * L / 2.) ** 2  * (1 - np.sin(kz * L) / (kz * L)))
            dos[:, omInd] += 1 / NSqr * np.sin(kz * (L / 2. - zArr))**2 * np.sin(kD * L / 2.) ** 2
    dosSummed = np.trapz(L**2 / (2. * np.pi * consts.c**2) * omArr[None, :] * dos[:, :], omArr, axis = 1)
    return dosSummed

def dosAnalytical(omega, zArr, eps, L):
    kzArr = allowedKsOm.findKs(L, omega, eps, "TE")
    kzArrDel = allowedKsOm.findKsDerivativeW(L, omega, eps, "TE")
    #print("kz = {}".format(kzArr))
    #print(" c^2 / omega delOm kz = {}".format(consts.c**2 / omega * kzArr * kzArrDel))
    kDArr = np.sqrt((eps - 1) * omega**2 / consts.c**2 + kzArr**2)
    NSqr = L / 4 * (eps * np.sin(kzArr * L / 2.)**2  * (1 - np.sin(kDArr * L) / (kDArr * L)) + np.sin(kDArr * L / 2.)**2 * (1 - np.sin(kzArr * L) / (kzArr * L)))
    func = np.sin(kzArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    return np.sum(1. / NSqr[None, :] * (1 -  consts.c**2 / omega * kzArr[None, :] * kzArrDel[None, :]) * func ** 2, axis=1)
    #return np.sum(1. / NSqr[None, :] * 1. / (1 +  consts.c**2 / omega * kzArr[None, :] * kzArrDel[None, :]) * func ** 2, axis=1)

def dosAnalyticalNoDerivative(omega, zArr, eps, L):
    kzArr = allowedKsOm.findKs(L, omega, eps, "TE")
    kDArr = np.sqrt((eps - 1) * omega**2 / consts.c**2 + kzArr**2)
    NSqr = L / 4 * (eps * np.sin(kzArr * L / 2.)**2  * (1 - np.sin(kDArr * L) / (kDArr * L)) + np.sin(kDArr * L / 2.)**2 * (1 - np.sin(kzArr * L) / (kzArr * L)))
    func = np.sin(kzArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    return np.sum(1. / NSqr[None, :] * func ** 2, axis=1)

def dosAnalyticalInt(zArr, omega, deltaOmega, eps, L):
    omArr = np.linspace(omega, omega + deltaOmega, 20)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        dosInt[:, wInd] = dosAnalytical(wVal, zArr, eps, L)

    return omega / (4. * np.pi * consts.c**2) * np.trapz(dosInt, omArr, axis=1)


def dosAnalyticalIntNoDerivative(zArr, omega, deltaOmega, eps, L):
    omArr = np.linspace(omega, omega + deltaOmega, 20)
    dosInt = np.zeros((len(zArr), len(omArr)))
    for wInd, wVal in enumerate(omArr):
        dosInt[:, wInd] = dosAnalyticalNoDerivative(wVal, zArr, eps, L)

    return omega / (4. * np.pi * consts.c**2) * np.trapz(dosInt, omArr, axis=1)



def plotDos(zArr, dos, dosAna, dosAnaWrong, L, omega, deltaOmega, eps):


    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    rho0 = (omega + deltaOmega / 2)**2 / (2. * np.pi**2 * consts.c ** 3)

    ax.plot(zArr, dos / deltaOmega / rho0, color='peru', lw=1., label = "DOS from Box")
    ax.plot(zArr, dosAna / deltaOmega / rho0, color='teal', lw=1., label = "DOS from Box", linestyle = '--')
    ax.plot(zArr, dosAnaWrong / deltaOmega / rho0, color='steelblue', lw=1., label = "DOS from Box", linestyle = '--')
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    #ax.axhline(0.5 * np.sqrt(eps), lw = 0.5, color = 'gray', zorder = -666)
    #ax.axhline(0.5 * np.sqrt(eps)**3, lw = 0.5, color = 'gray', zorder = -666)

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    #ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    #ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$z[m]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.savefig("./savedPlots/dosPedestrianDielectric.png")

def plotDosComp(zArr, dos, dosInt, L, omega, deltaOmega, eps):

    dosAna = dosAnalyticalInt(zArr, omega, deltaOmega, eps, L)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    rho0 = (omega + deltaOmega / 2)**2 / (2. * np.pi**2 * consts.c ** 3)

    #ax.plot(zArr, dos / deltaOmega / rho0, color='peru', lw=1., label = "DOS from Box")
    ax.plot(zArr, dosInt / deltaOmega / rho0, color='coral', lw=1., label = "DOS from Box")
    ax.plot(zArr, dosAna / deltaOmega / rho0, color='teal', lw=1., label = "DOS from Box", linestyle = '--')
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps), lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**3, lw = 0.5, color = 'gray', zorder = -666)

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    #ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    #ax.set_xticklabels([r"$-\frac{L}{500}$", r"$0$", r"$\frac{L}{500}$"])

    ax.set_xlabel(r"$z[m]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.savefig("./savedPlots/dosPedestrianDielectric.png")


def pedestrianMainDielectric():

    L = .1
    eps = 2.
    omega = .3 * 1e11
    deltaOmega = omega / 10.

    kBound = np.sqrt(eps) * (omega + deltaOmega) / consts.c
    nBound = math.ceil(kBound * L / (2. * np.pi))
    kxInds = np.arange(- nBound, nBound)
    kxArr = kxInds * 2. * np.pi / L
    kyArr = kxArr
    zArr = np.linspace(0., L / 2., 500)

    dos = dosPedestrian(kxArr, kyArr, zArr, omega, deltaOmega, eps, L)
    #dosOmInt = dosPedestrianOmInt(zArr, omega, deltaOmega, eps, L)
    #dosInt = dosPedestrianInt(zArr, omega, deltaOmega, eps, L)
    dosAna = dosAnalyticalInt(zArr, omega, deltaOmega, eps, L)
    dosAnaWrong = dosAnalyticalIntNoDerivative(zArr, omega, deltaOmega, eps, L)
    plotDos(zArr, dos, dosAna, dosAnaWrong, L, omega, deltaOmega, eps)