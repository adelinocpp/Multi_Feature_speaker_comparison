#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:27:33 2022

@author: adelino
"""
import numpy as np
import scipy
from sklearn.neighbors import KernelDensity

# -----------------------------------------------------------------------------
def vector_entropy(x):
    X = x.reshape(-1,1)
    xStd = np.std(x)
    xMean = np.mean(x)
    band = 1.06*xStd*(len(x)**(-1/5))
    Nden = 250
    h = 7*xStd/(Nden+1)
    x_dom = np.linspace(xMean-3.5*xStd, xMean+3.5*xStd, num=Nden)
    kde = KernelDensity(kernel='gaussian',bandwidth=band).fit(X)
    prob_mtx = np.exp(kde.score_samples(x_dom.reshape(-1,1)))
    f = np.multiply(prob_mtx,np.log(prob_mtx + np.finfo(float).eps))
    I_simp = (h/3) * (f[0] + 2*np.sum(f[:Nden-2:2]) \
            + 4*np.sum(f[1:Nden-1:2]) + f[Nden-1])
    return -I_simp
# -----------------------------------------------------------------------------
def mag_phase_filter(f_x, f_m, f_p):
    n_x = len(f_x)
    n_m = len(f_m)
    if (n_m < n_x):
        f_m = np.append(f_m,np.flip(f_m))
        f_p = np.append(f_p,np.flip(-f_p))
    
    y_m_f = np.multiply(f_m,np.exp(1j*f_p))
    return np.real(scipy.fft.ifft(np.multiply(f_x,y_m_f)))
# -----------------------------------------------------------------------------
def dct_aps(x, K=0):
    N = x.shape[0]
    if (K == 0) or (K > N):
        K = N
    c = np.zeros((K,1))
    Mconst = np.sqrt(2/K)
    for k in range(0,K):
        if (k == 0):
            beta = 1/np.sqrt(2)
        else:
            beta = 1
        kSum = [x[n]*np.cos(k*np.pi/N*(0.5+n)) for n in range(0,N)]
        c[k] = Mconst*beta*np.sum(kSum)
    return c