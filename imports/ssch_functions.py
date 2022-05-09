#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:21:36 2022

@author: adelino
"""
import numpy as np
from .build_filters import build_bark_retang_filters
from .convert_frequency import bark2hertz, hertz2bark
from .signal_process_util import simpson_integral

def ssch_histogram(x_fft, sr, nfilts, timeWin = 0.025, min_freq=0,max_freq=0):
    n_FFT = len(x_fft)
    if (max_freq == 0):
        max_freq = 0.5*sr

    min_mel = hertz2bark(min_freq)
    nyq_mel = hertz2bark(max_freq) - min_mel
    step_mel = nyq_mel/(nfilts+1)
    bin_hertz = np.array([bark2hertz(min_mel + (i+1)*step_mel) for i in range(0,nfilts)])
    ssch_filter = build_bark_retang_filters(2*n_FFT,sr,nfilts=nfilts)
    
    vf = np.array([i*0.5*sr/n_FFT for i in range(0,n_FFT)])
    x_hist = np.zeros((nfilts+1,1))
    
    for k in range (0,nfilts):
        x_freq = np.multiply(np.abs(x_fft),ssch_filter[k,:])
        PowK = simpson_integral(vf,x_freq)
        Ck = np.sum(np.multiply(vf,x_freq))/np.sum(x_freq)
        n = np.nonzero(Ck > bin_hertz)[0]
        if (len(n) > 0):
            x_hist[n[-1]+1] = x_hist[n[-1]+1] + np.log(PowK)
        elif(Ck < bin_hertz[0]):
            x_hist[0] = x_hist[0] + np.log(PowK)
    return x_hist