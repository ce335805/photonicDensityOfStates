import numpy as np
import scipy.integrate as integrate

def complexSqrt(R, I):

    r = np.sqrt(R**2 + I**2)
    sqrR = np.sqrt(r) * (R + r) / (np.sqrt((R * r)**2 + I**2))
    sqrI = np.sqrt(r) * I / (np.sqrt((R * r)**2 + I**2))

    return (R, I)

def complexSinSqr(R, I):
    term1 = np.cosh(I)**2 * np.sin(R)**2
    term2 = - np.sinh(R)**2 * np.cos(I)**2
    term3 = 2j * np.sin(R) * np.cos(R) * np.sinh(I) * np.cosh(I)

    return term1 + term2 + term3

def complexSin(R, I):
    return np.cosh(I) * np.sin(R) - 1j * np.cos(R) * np.sinh(I)

def complexCos(R, I):
    return np.cos(R) * np.cos(I) + 1j * np.sin(R) * np.sin(I)

def complexIntegrand(kI, kR, z, L, omega, eps, c):

    kTSqrR = (eps - 1) * omega**2 / c**2 + kR**2 + kI**2
    kTSqrI = 2. * kR * kI

    kTR, kTI = complexSqrt(kTSqrR, kTSqrI)

    sin1term = complexSinSqr(kR * (L / 2 - z), kI * (L / 2 - z))
    sin2term = complexSinSqr(kTR * L / 2., kTI * L / 2.)
    sin3term = complexSinSqr(kR * L / 2., kI * L / 2.)
    term = sin1term * sin2term / (sin3term + sin2term)

    poleTerm1 = (kTR + 1j * kTI) * complexSin(kTR * L / 2., kTI * L / 2.) * complexCos(kR * L / 2., kI * L / 2.)
    poleTerm2 = (kR + 1j * kI) * complexSin(kR * L / 2., kI * L / 2.) * complexCos(kTR * L / 2., kTI * L / 2.)

    poleTerm = 1. / (poleTerm1 - poleTerm2)

    return term * poleTerm


def performComplexIntegral(z, L, omega, epsilon, c):

    infP = 1e1
    infM = -1e1

    int1 = integrate.quad(complexIntegrand, infP, infM, args=(0., z, L, omega, epsilon, c))

    print("int at 0 = {}".format(int1))

