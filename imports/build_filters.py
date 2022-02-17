#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:31:13 2022

@author: adelino
"""
import numpy as np
from .convert_frequency import hertz2bark, hertz2mel

# -----------------------------------------------------------------------------
def build_bark_filters(nfft,sr,nfilts=0,width=1,min_freq=0,max_freq=0):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_bark = hertz2bark(min_freq)
    nyq_bark = hertz2bark(max_freq) - min_bark
    if (nfilts == 0):
        nfilts = np.ceil(nyq_bark) + 1
    h_fft = int(0.5*nfft)
    wts = np.zeros((nfilts, h_fft))

    step_barks = nyq_bark/(nfilts-1)
    bin_barks = np.array([hertz2bark(i*sr/nfft) for i in range(0,h_fft)])
    limits = np.empty((2,h_fft))
    for i in range(0,nfilts):
        f_bark_mid = min_bark + i*step_barks
        limits[0,:] = -2.5*(bin_barks - f_bark_mid - 0.5)
        limits[1,:] = (bin_barks - f_bark_mid + 0.5)
        wts[i,:] = np.power(10, np.minimum(np.zeros((h_fft,)), np.min(limits,axis=0)/width))
    return wts
# -----------------------------------------------------------------------------
def build_bark_filters(nfft,sr,nfilts=0,width=1,min_freq=0,max_freq=0):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_mel = hertz2mel(min_freq)
    nyq_mel = hertz2mel(max_freq) - min_mel
    if (nfilts == 0):
        nfilts = np.ceil(4.6*np.log10(sr))
        # nfilts = np.ceil(nyq_mel) + 1
    h_fft = int(0.5*nfft)
    wts = np.zeros((nfilts, h_fft))

    step_mel = nyq_mel/(nfilts-1)
    # bin_barks = np.array([hertz2bark(i*sr/nfft) for i in range(0,h_fft)])
    
    for i in range(0,nfilts):
        f_mid = min_mel + (i+1)*step_mel
        f_ini = f_mid - i*step_mel
        f_fim = f_mid + i*step_mel
        # TODO: Fazer a função triangular no dominio da frequencia
        # limits[0,:] = -2.5*(bin_barks - f_bark_mid - 0.5)
        # limits[1,:] = (bin_barks - f_bark_mid + 0.5)
        # wts[i,:] = np.power(10, np.minimum(np.zeros((h_fft,)), np.min(limits,axis=0)/width))
    return wts