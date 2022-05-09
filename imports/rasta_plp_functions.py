#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:27:43 2022

@author: adelino
"""
import numpy as np
import numpy.matlib as npm
import scipy
from scipy import signal
from .convert_frequency import bark2hertz, hertz2bark
from .signal_process_util import levinson_aps

# -----------------------------------------------------------------------------
def postaud(x,fmax,broaden=0):
    nbands,nframes = x.shape

    # equal loundness weights stolen from rasta code
    # eql = [0.000479 0.005949 0.021117 0.044806 0.073345 0.104417 0.137717 ...
    #        0.174255 0.215590 0.263260 0.318302 0.380844 0.449798 0.522813
    #        0.596597];
    nfpts = nbands + 2*broaden
    bandcfhz = [bark2hertz(i*hertz2bark(fmax)/(nfpts-1)) for i in range(0, nfpts)]    
    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[broaden:(nfpts-broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = np.power(bandcfhz,2)
    ftmp = fsq + 1.6e5
    eql = np.multiply( np.power(np.divide(fsq,ftmp),2), \
        np.divide(fsq+1.44e6,ftmp+ 9.61e6) )
    # eql = ((fsq./ftmp).^2) .* ((fsq + 1.44e6)./(fsq + 9.61e6));

    # weight the critical bands
    eql.shape = (eql.shape[0],1)
    z = np.multiply(npm.repmat(eql, 1,nframes),x)

    # cube root compress
    z = np.power(z,0.33)

    # replicate first and last band (because they are unreliable as calculated)
    if (broaden == 1):
        y = np.concatenate((z[0:1,:],z,z[-1:,:]),axis=0)
        #z[[1,1:nbands,nbands],:]
    else:      
        y = np.concatenate((z[1:2,:],z[1:-1,:],z[-2:-1,:]),axis=0)
        # y = z[[2,2:(nbands-1),nbands-1],:]
    return y

# -----------------------------------------------------------------------------
def lifter(x, lift=0.6, invs=0):
    ncep = x.shape[0]
    if (lift == 0):
        y = x
    else:
        if (lift > 0):
            if (lift > 10):
                print('Unlikely lift exponent of {:} (did you mean -ve?)'.format(lift))
            # TODO: Conferir esta equação
            liftwts = np.concatenate((np.array([1]), \
                                  np.array([i**lift for i in range(1,ncep)])))
        elif (lift < 0):
            L = -lift
            if (L != np.round(L)):
                print('HTK liftering value {:} must be integer'.format(L))
            liftwts = np.concatenate((np.array([1]), \
                            np.array([1+0.5*L*np.sin(i*np.pi/L) for i in range(1,ncep)])))
 #           liftwts = [1, (1+L/2*sin([1:(ncep-1)]*pi/L))];
        if (invs == 1):
            liftwts = 1./liftwts
        y = np.matmul(np.diag(liftwts),x)
    return y
# -----------------------------------------------------------------------------
def lpc2cep(a,nout=0):
    [nin, nframes] = a.shape
    order = nin - 1;
    if (nout == 0):
        nout = order + 1;
    c = np.zeros((nout,nframes))
    c[0,:] = -np.log(a[0,:])
    a = a/a[0,:]
    for n in range(1,nout):
        Soma = 0
        for m in range(1,n):
            Soma += (n - m) * a[m,:] * c[n - m,:]
        c[n,:] = -(a[n,:] + Soma/n);
    return c
# -----------------------------------------------------------------------------
def do_lpc(spec, order=8):
    [nbands, nframes] = spec.shape
    spec_c = spec[1:-1,:]
    x = np.concatenate((spec,np.flipud(spec_c)),axis=0)
    x = np.real(scipy.fft.ifft(x,axis=0))
    x = x[:nbands,:]
    y = np.zeros((order+1,nframes))
    for idx in range(0,nframes):
        a, e = levinson_aps(x[:,idx], order)
        y[:,idx] = a/e
        # y[:,idx] = lpc2cep(a/e,order+1)
    return y
# -----------------------------------------------------------------------------
def rasta_filter(spec):
    numer = np.array(range(-2,3))
    numer = -numer /np.sum(numer*numer);
    denom = np.array([1, -0.94]);
    zi = signal.lfilter_zi(numer, [1])
    y, z = signal.lfilter(numer, [1], spec[:4],axis=0,zi=0*zi)
    y0 = y*0
    y1 = signal.lfilter(numer, denom, spec[4:],axis=0,zi=z)
    return np.concatenate((y0,y1[0]),axis=0)
