#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:28:38 2022

@author: adelino
"""
import numpy as np

# -----------------------------------------------------------------------------
def erb2hertz(erb_freq):
    vu = np.array([6.23e-6, 93.39e-3, 28.52])
    vp = np.sort(np.roots(vu))             # p=[-14678.5 -311.9]
    vd = 1e-6*(6.23*(vp[1]-vp[0]))    # d=0.0895
    vc = vp[0]                       # c=-14678.5
    vk = vp[0] - vp[0]**2/vp[1]         # k=676170.4
    vh = vp[0]/vp[1]                  # h=47.06538

    frq = np.sign(erb_freq)*(vk/np.max([vh - np.exp(vd*np.abs(erb_freq)),0]) + vc)
    bnd = np.polyval(vu,abs(frq))
    return frq, bnd
# -----------------------------------------------------------------------------
def hertz2erb(hertz_freq):
    u = np.array([6.23e-6, 93.39e-3, 28.52])
    p = np.sort(np.roots(u))        # p=[-14678.5 -311.9]
    a = 1e6/(6.23*(p[1]-p[0]));       # a=11.17
    c = p[0]                        # c=-14678.5
    k = p[0] - p[0]**2/p[1]         # k=676170.42
    h = p[0]/p[1]                    # h=47.065

    g = np.abs(hertz_freq)
    # erb=11.17268*sign(frq).*log(1+46.06538*g./(g+14678.49));
    erb = a*np.sign(hertz_freq)*np.log(h-k/(g-c))
    bnd = np.polyval(u,g)
    return erb, bnd

# -----------------------------------------------------------------------------
def bark2hertz(bark_freq):
    return 600 * np.sinh(bark_freq/6)
# -----------------------------------------------------------------------------
def hertz2bark(hertz_freq):
    return 6*np.arcsinh(hertz_freq/600)
# -----------------------------------------------------------------------------
def hertz2mel(hertz_freq,Slaney=False):
    if (Slaney):
        f_0 = 0 # 133.33333;
        f_sp = 200/3 # 66.66667;
        brkfrq = 1000
        brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
        logstep = np.exp(np.log(6.4)/27); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
        if (hertz_freq < brkfrq):
            mb = (hertz_freq - f_0)/f_sp;
        else:
            mb =  brkpt+(np.log(hertz_freq/brkfrq))/np.log(logstep);
    else:
        k=1000/np.log(1+1000/700) # 1127.01048
        af = np.abs(hertz_freq)
        mb = np.sign(hertz_freq)*np.log(1+af/700)*k
    return mb
# -----------------------------------------------------------------------------
def mel2hertz(mel_freq,Slaney=False):
    if (Slaney):
        f_0 = 0 # 133.33333;
        f_sp = 200/3 # 66.66667;
        brkfrq = 1000
        brkpt  = (brkfrq - f_0)/f_sp # starting mel value for log region
        logstep = np.exp(np.log(6.4)/27) # the magic 1.0711703 which is the ratio 
        # needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the 
        # ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz 
        # (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        if (mel_freq < brkpt):
            mb = f_0 + f_sp*mel_freq
        else:
            mb = brkfrq*np.exp(np.log(logstep)*(mel_freq-brkpt))
    else:
        k=1000/np.log(1+1000/700) # 1127.01048
        am = np.abs(mel_freq)
        mb = 700*np.sign(mel_freq)*(np.exp(am/k)-1)
    return mb
