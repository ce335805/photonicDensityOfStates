import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import math
import matplotlib.cm as cm
from scipy.integrate import quad

import dosSurface as surf

fontsize = 10


def freeRho(w):
    return w**2 / (const.c**3 * np.pi**2)


def cavityRho2(w, h):

    bound = h * w / (np.pi * const.c)

    if(bound < 1):
        return 0.

    boundSum = math.floor(0.5 * (math.floor(bound) - 1))

    sum = 1. + boundSum + \
          0.5 * boundSum * (boundSum + 1) + \
          1./6. * boundSum * (boundSum + 1.) * (2. * boundSum + 1.)

    prefac = np.pi / (2  * h**3 * w)

    return 1. * (w / (2. * np.pi * h * const.c**2) * math.floor(bound) + prefac * sum)

def cavityRhoDiscrete(N, h):
    w = N * np.pi * const.c / h
    rhoFree = w ** 2 / (np.pi**2 * const.c**3)
    term1 = (N + 1)/(2 * N)
    term2 = (N + 1) / 2 + 0.5 * (N - 1) / 2 * (N + 1) / 2 + 1. / 6. * (N - 1) / 2 * (N + 1) / 2 * N
    return rhoFree * (term1 + 1 / (2 * N ** 3) * term2)

def integrateAsOfOmega(omegaArr, rhoArr):

    sumArr = np.zeros(len(rhoArr))
    for ind in range(len(rhoArr)):
        sumArr[ind] = np.sum(rhoArr[:ind]) * (omegaArr[1] - omegaArr[0])
    return sumArr

def plotDensityOfStates(hArr, wArr, cavityRhoArr):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig.set_size_inches(2.5, 1.7)

    cmapPink = cm.get_cmap('pink')

    for hInd, h in enumerate(hArr):
        color = cmapPink(0.7 - hInd / 5.)
        ax.plot(wArr / 1e12, cavityRhoArr[hInd, :] / (1e-12 * wArr) ** 2, color=color, linewidth = 1.,
                label = "d = {0:.2g}mm".format(h * 1e3))

    # ax.plot(wArr / 1e12, rhoFreeArr / (1e-12 * wArr)**2, linestyle = '--', color='black', linewidth = 2.)

    ax.set_ylabel(r'$\rho / \omega^2 [\dots]$')
    ax.set_xlabel(r'$\omega$[THz]')

    legend = ax.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(-0.4, 1.), edgecolor='black', ncol=2)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(0)

    plt.savefig('./rho.png', format='png', bbox_inches='tight', dpi=600)

def plotIntegratedDensityOfStates(hArr, wArr, cavityRhoArr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(2.5, 1.7)


    cmapPink = cm.get_cmap('pink')

    for hInd, h in enumerate(hArr):
        color = cmapPink(0.7 - hInd / 5.)
        ax.plot(wArr / 1e12, cavityRhoArr[hInd, :] / (1e-12 * wArr) ** 2 * 1e-12, color=color, linewidth = 1.)

    # ax.plot(wArr / 1e12, rhoFreeArr / (1e-12 * wArr)**2, linestyle = '--', color='black', linewidth = 2.)

    ax.set_ylabel(r"$\int \rho / \omega'^2 d\omega' [\dots]$")
    ax.set_xlabel(r'$\omega$[THz]')

    plt.savefig('./rhoInegrated.png', format='png', bbox_inches='tight', dpi=600)

def integrand(z, a):
    return z**2 * np.exp(- a * z)

def main():
    #surf.plotrhoAsOfz()
    surf.plotrhoIntAsOfZ()
    #surf.plotRhoZs()
    #surf.plotRhoFreqs()

    exit()

    NX = 100
    xArr = np.linspace(0.1, 1., NX)
    #NInt = 10000
    #zArr = np.linspace(0., 1000., NInt, endpoint=False)
    #ones = np.ones(NX)
    #intArr = np.outer(zArr**2, ones) * np.exp(-np.outer(zArr, xArr))
    #int = np.sum(intArr, axis = 0) * (zArr[1] - zArr[0])
    int = np.zeros(NX)
    for xInd, xVal in enumerate(xArr):
        int[xInd] = quad(integrand, 0., np.inf, args=(xVal))[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(2.5, 1.7)
    ax.plot(xArr, 2 / xArr**3)
    ax.plot(xArr, int, ls = '--')
    plt.savefig('./savedPlots/testIntegral.png', format='png', bbox_inches='tight', dpi=600)


    exit()

    wArr = np.linspace(1., 1e14 * 0.5, 10000)
    hArr = np.array([1e-5 * 3., 1e-4, 1e-3, 1e-2])

    rhoFreeArr = np.zeros(len(wArr))
    cavityRhoArr = np.zeros((len(hArr), len(wArr)))

    for wInd, w in enumerate(wArr):
        rhoFreeArr[wInd] = freeRho(w)
        for hInd, h in enumerate(hArr):
            cavityRhoArr[hInd, wInd] = cavityRho2(w, h)

    plotDensityOfStates(hArr, wArr, cavityRhoArr)

    integratedRhoArr = np.zeros((len(hArr), len(wArr)))

    wArrL = np.linspace(1., 1e14, 10000)
    cavityRhoArr = np.zeros((len(hArr), len(wArrL)))

    for wInd, w in enumerate(wArrL):
        for hInd, h in enumerate(hArr):
            cavityRhoArr[hInd, wInd] = cavityRho2(w, h)

    for hInd, h in enumerate(hArr):
        integratedRhoArr[hInd, :] = integrateAsOfOmega(wArrL, cavityRhoArr[hInd, :])


    plotIntegratedDensityOfStates(hArr, wArrL, integratedRhoArr)


main()