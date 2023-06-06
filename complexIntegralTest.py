import numpy as np
import math
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
from matplotlib import gridspec

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

def kPoints(cutoff, L):
    discreteN = np.arange(1, math.floor(cutoff * L / (2. * np.pi)))
    return discreteN * 2. * np.pi / L

def sumNumerically(x, cutoff, L):
    ks = kPoints(cutoff, L)
    print("Summing over {} k-Points".format(len(ks)))
    sumArr = 1. / L * np.sin(ks * x)
    return np.sum(sumArr)

def intFunc(k, x):
    return 1. / (2. * np.pi) * np.sin(k * x)
#gives correct result in limit L -> infinity
def sumAsSimpleIntegral(x, cutoff):
    intRes = integrate.quad(intFunc, 0, cutoff, args=(x))
    return intRes[0]

def complexIntFunc(k, x, L, eps, Lam):
    A = np.sin(eps * x) * np.sin(eps * L) * np.cosh(k * x) * np.cosh(k * L) \
        + np.cos(eps * x) * np.cos(eps * L) * np.sinh(k * x) * np.sinh(k * L)

    B = np.sin(eps * L)**2 * np.cosh(k * L)**2 + np.cos(eps * L)**2 * np.sinh(k * L)**2

    C = np.sin(Lam * x) * np.sin(Lam * L) * np.cosh(k * x) * np.cosh(k * L) \
        + np.cos(Lam * x) * np.cos(Lam * L) * np.sinh(k * x) * np.sinh(k * L)

    D = np.sin(Lam * L)**2 * np.cosh(k * L)**2 + np.cos(Lam * L)**2 * np.sinh(k * L)**2

    return 1. / (2. * np.pi) * C / D - A / B

def sumComplexIntegral(x, L, eps, cutoff):
    res = integrate.quad(complexIntFunc, - 10. / L, 10. / L, args=(x, L, eps, cutoff))
    return res[0]

def plotComplexIntFunc(x, L, eps, cutoff):
    kArr = np.linspace(-10. / L, 10. / L, 1000)

    intF = complexIntFunc(kArr, x, L, eps, cutoff)

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(kArr, intF, color='indianred', lw=0.8)

    plt.savefig("./savedPlots/complexIntFuncTest.png")


def testMain():


    L = 10000.
    cutoff = 100
    x = 5.

    resultSum = sumNumerically(x, cutoff, L)
    print("Result Sum = {}".format(resultSum))

    resultInt = sumAsSimpleIntegral(x, cutoff)
    print("Result Int = {}".format(resultInt))

    resultCInt = sumComplexIntegral(x, L, 1e-7, cutoff)
    print("Result CInt = {}".format(resultCInt))

    print("CInt / Sum = {}".format(resultSum / resultCInt))

    plotComplexIntFunc(x, L, 1e-7, cutoff)


testMain()
