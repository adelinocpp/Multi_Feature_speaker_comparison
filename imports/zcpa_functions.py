#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:57:10 2022

@author: adelino
"""
import numpy as np
from .convert_frequency import hertz2bark, bark2hertz, hertz2mel, mel2hertz, \
                            hertz2erb, erb2hertz
from .build_filters import build_mel_triang_filters, build_erb_triang_filters, \
                        build_bark_triang_filters
from .signal_process_util import mag_phase_filter
from .teocc_functions import shift              
# -----------------------------------------------------------------------------
def zcpa_histogram(x_fft, sr, nfilts, timeWin = 0.025, min_freq=0,max_freq=0, typefilter='bark'):
    n_FFT = len(x_fft)
    if (max_freq == 0):
        max_freq = 0.5*sr
        
    if (typefilter.upper() == 'MEL'):
        min_mel = hertz2mel(min_freq)
        nyq_mel = hertz2mel(max_freq) - min_mel
        step_mel = nyq_mel/(nfilts+1)
        bin_hertz = np.array([mel2hertz(min_mel + (i+1)*step_mel) for i in range(0,nfilts)])
        zcpa_filter = build_mel_triang_filters(n_FFT,sr,nfilts=nfilts)
        
    elif (typefilter.upper() == 'ERB'):
        min_mel = hertz2erb(min_freq)
        nyq_mel = hertz2erb(max_freq) - min_mel
        step_mel = nyq_mel/(nfilts+1)
        bin_hertz = np.array([erb2hertz(min_mel + (i+1)*step_mel) for i in range(0,nfilts)])
        zcpa_filter = build_erb_triang_filters(n_FFT,sr,nfilts=nfilts)
    else:
        min_mel = hertz2bark(min_freq)
        nyq_mel = hertz2bark(max_freq) - min_mel
        step_mel = nyq_mel/(nfilts+1)
        bin_hertz = np.array([bark2hertz(min_mel + (i+1)*step_mel) for i in range(0,nfilts)])
        zcpa_filter = build_bark_triang_filters(n_FFT,sr,nfilts=nfilts)
    
    vt = np.array([i*timeWin/n_FFT for i in range(0,n_FFT)])
    x_hist = np.zeros((nfilts+1,1))
    
    for k in range (0,nfilts):
        x_time = mag_phase_filter(x_fft,zcpa_filter[k,:], np.zeros(zcpa_filter[k,:].shape))
        zcs = np.multiply(x_time,shift(x_time, -1, 0))
        idx = np.nonzero(zcs < 0)[0]
        
        for i in range(0,len(idx)-2):
            idxIni = idx[i]
            idxFim = idx[i+1]
            Fk = 1/(vt[idxFim] - vt[idxIni])
            Pk =  np.max(np.abs(x_time[idxIni:idxFim]))
            n = np.nonzero(Fk > bin_hertz)[0]
            if (len(n) > 0):
                x_hist[n[-1]+1] = x_hist[n[-1]+1] + np.log(1 + Pk)
            elif(Fk < bin_hertz[0]):
                x_hist[0] = x_hist[0] + np.log(1 + Pk)
    return x_hist