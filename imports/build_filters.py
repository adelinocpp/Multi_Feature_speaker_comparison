#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:31:13 2022

@author: adelino
"""
import numpy as np
from .convert_frequency import hertz2bark, bark2hertz, hertz2mel, mel2hertz, \
                            hertz2erb, erb2hertz
import scipy

# -----------------------------------------------------------------------------
def gammatone_filter(cfs,f_points,f_Order = 4,align=False,onlyhalfMag=False):
    if (f_Order < 0):
        f_Order = 4     # filter order
    nPts = len(f_points)       # gammatone filter length at least 128 ms
    b = 1.019*24.7*(4.37*cfs/1000 + 1) # rate of decay or bandwidth
    fs = 2*f_points[-1]*(nPts)/(nPts-1)
    
    gt = np.zeros((nPts,))         # Initialise IR
    tc = 0  # Initialise time lead
    phase = 0;

    tpt = (2*np.pi)/fs
    gain = ((1.019*b*tpt)**f_Order)/6; # based on integral of impulse

    tmp_t = np.array([n/fs for n in range(0,2*nPts)])
    #tmp_t = np.array([n/fs for n in range(0,nPts)])

    if (align):
        tc = (f_Order-1)/(2*np.pi*b)
        phase = -2*np.pi*cfs*tc
    
    t_exp = np.multiply(np.power(tmp_t,f_Order-1),np.exp(-2*np.pi*b*tmp_t))
    gt = gain*(fs**3)*np.multiply(t_exp,np.cos(2*np.pi*cfs*tmp_t + phase))
    
    fGT = scipy.fft.fft(gt)
    if (onlyhalfMag):
        return np.abs(fGT[:nPts])
    else:
        Mag = np.abs(fGT)
        Phase = np.angle(fGT)
        return Mag, Phase
# -----------------------------------------------------------------------------
def build_bark_plp_filters(nfft,sr,nfilts=0,width=1,min_freq=0,max_freq=0):
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
def build_mel_tiang_filters(nfft,sr,nfilts=0,min_freq=0,max_freq=0,Slaney=False):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_mel = hertz2mel(min_freq,Slaney)
    nyq_mel = hertz2mel(max_freq,Slaney) - min_mel
    if (nfilts == 0):
        nfilts = np.ceil(4.6*np.log10(sr))
        # nfilts = np.ceil(nyq_mel) + 1
    h_fft = int(0.5*nfft)
    wts = np.zeros((nfilts, h_fft))
    step_mel = nyq_mel/(nfilts+1)
    bin_hertz = np.array([i*sr/nfft for i in range(0,h_fft)])
    limits = np.empty((2,h_fft))
    for i in range(0,nfilts):
        f_mid = mel2hertz(min_mel + (i+1)*step_mel,Slaney)
        f_ini = mel2hertz(min_mel + i*step_mel,Slaney)
        f_fim = mel2hertz(min_mel + (i+2)*step_mel,Slaney)
        
        limits[0,:] = (bin_hertz - f_fim)/(f_mid-f_fim)
        limits[1,:] = (bin_hertz - f_ini)/(f_mid-f_ini)
        if (Slaney):
            kMult = 2/(f_fim - f_ini)
        else:
            kMult = 1
        wts[i,:] = kMult*np.maximum(np.zeros((h_fft,)), np.min(limits,axis=0))
    return wts
# -----------------------------------------------------------------------------
def build_bark_tiang_filters(nfft,sr,nfilts=0,min_freq=0,max_freq=0):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_mel = hertz2bark(min_freq)
    nyq_mel = hertz2bark(max_freq) - min_mel
    if (nfilts == 0):
        nfilts = np.ceil(4.6*np.log10(sr))
        # nfilts = np.ceil(nyq_mel) + 1
    h_fft = int(0.5*nfft)
    wts = np.zeros((nfilts, h_fft))
    step_mel = nyq_mel/(nfilts+1)
    bin_hertz = np.array([i*sr/nfft for i in range(0,h_fft)])
    limits = np.empty((2,h_fft))
    for i in range(0,nfilts):
        f_mid = bark2hertz(min_mel + (i+1)*step_mel)
        f_ini = bark2hertz(min_mel + i*step_mel)
        f_fim = bark2hertz(min_mel + (i+2)*step_mel)
        
        limits[0,:] = (bin_hertz - f_fim)/(f_mid-f_fim)
        limits[1,:] = (bin_hertz - f_ini)/(f_mid-f_ini)
        wts[i,:] = np.maximum(np.zeros((h_fft,)), np.min(limits,axis=0))
    return wts
# -----------------------------------------------------------------------------
def build_erb_gamma_filters(nfft,sr,nfilts=0,min_freq=0,max_freq=0,halfMag=True):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_erb, _ = hertz2erb(min_freq)
    nyq_erb, _ = hertz2erb(max_freq) - min_erb
    if (nfilts == 0):
        nfilts = np.ceil(4.6*np.log10(sr))
        # nfilts = np.ceil(nyq_mel) + 1
    h_fft = int(0.5*nfft)
    
    step_erb = nyq_erb/(nfilts+1)
    bin_hertz = np.array([i*sr/nfft for i in range(0,h_fft)])
    # limits = np.empty((2,h_fft))
    if (halfMag):
        wts = np.zeros((nfilts, h_fft))
        for i in range(0,nfilts):
            f_mid, _ = erb2hertz(min_erb + (i+1)*step_erb)
            wts[i,:] = gammatone_filter(f_mid,bin_hertz,onlyhalfMag=halfMag)
        return wts
    else:
        wts = np.zeros((nfilts, h_fft))
        pts = np.zeros((nfilts, h_fft))
        for i in range(0,nfilts):
            f_mid, _ = erb2hertz(min_erb + (i+1)*step_erb)
            m, p = gammatone_filter(f_mid,bin_hertz,onlyhalfMag=halfMag)
            wts[i,:] = m
            pts[i,:] = p
        return wts, pts
# -----------------------------------------------------------------------------    
def build_mel_gamma_filters(nfft,sr,nfilts=0,min_freq=0,max_freq=0,halfMag=True):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_mel = hertz2mel(min_freq)
    nyq_mel = hertz2mel(max_freq) - min_mel
    if (nfilts == 0):
        nfilts = np.ceil(4.6*np.log10(sr))
        # nfilts = np.ceil(nyq_mel) + 1
    h_fft = int(0.5*nfft)
    
    step_mel = nyq_mel/(nfilts+1)
    bin_hertz = np.array([i*sr/nfft for i in range(0,h_fft)])
    # limits = np.empty((2,h_fft))
    if (halfMag):
        wts = np.zeros((nfilts, h_fft))
        for i in range(0,nfilts):
            f_mid, _ = mel2hertz(min_mel + (i+1)*step_mel)
            wts[i,:] = gammatone_filter(f_mid,bin_hertz,onlyhalfMag=halfMag)
        return wts
    else:
        wts = np.zeros((nfilts, h_fft))
        pts = np.zeros((nfilts, h_fft))
        for i in range(0,nfilts):
            f_mid, _ = mel2hertz(min_mel + (i+1)*step_mel)
            m, p = gammatone_filter(f_mid,bin_hertz,onlyhalfMag=halfMag)
            wts[i,:] = m
            pts[i,:] = p
        return wts, pts
# -----------------------------------------------------------------------------