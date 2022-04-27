#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv



def bessel(v, X):
    return ((1j**(-v))*jv(v,1j*X)).real
# -----------------------------------------------------------------------------
class estnoisem_frame_param:
    def __init__(self, p_init, nFFT2):
        self.t      = 0
        self.p      = p_init         # smoothed power spectrum
        self.ac     = 1              # correction factor (9)
        self.sn2    = self.p         # estimated noise power
        self.pb     = self.p               # smoothed noisy speech power (20)
        self.pb2    = self.pb**2
        self.pminu  = self.p
        self.actmin = np.array(np.ones(nFFT2) * np.Inf)   # Running minimum estimate
        self.actminsub = np.array(np.ones(nFFT2) * np.Inf)          # sub-window minimum estimate
        self.subwc  = 0
        self.actbuf = 0  # buffer to store subwindow minima
        self.ibuf   = 0
        self.lminflag = np.zeros(nFFT2)      # flag to remember local minimum
        
    def set_param(self,t,p,ac,sn2,pb,pb2,pminu,actmin, actminsub, subwc, actbuf, ibuf, lminflag):
        self.t      = t
        self.p      = p
        self.ac     = ac 
        self.sn2    = sn2
        self.pb     = pb
        self.pb2    = pb2
        self.pminu  = pminu
        self.actmin = actmin
        self.actminsub = actminsub
        self.subwc  = subwc
        self.actbuf = actbuf
        self.ibuf   = ibuf
        self.lminflag = lminflag
# -----------------------------------------------------------------------------        
def estnoisem_frame(pSpectrum,hop_length, param_em):
    nFFT2 = pSpectrum.shape[0]          # number of frames and freq bins
    x = np.zeros((nFFT2,))            # initialize output arrays

    # default algorithm constants
    taca= 0.0449    # smoothing time constant for alpha_c = -hop_length/log(0.7) in equ (11)
    tamax= 0.392    # max smoothing time constant in (3) = -hop_length/log(0.96)
    taminh= 0.0133    # min smoothing time constant (upper limit) in (3) = -hop_length/log(0.3)
    tpfall= 0.064   # time constant for P to fall (12)
    tbmax= 0.0717   # max smoothing time constant in (20) = -hop_length/log(0.8)
    qeqmin= 2.0       # minimum value of Qeq (23)
    qeqmax= 14.0      # max value of Qeq per frame
    av= 2.12             # fudge factor for bc calculation (23 + 13 lines)
    td= 1.536       # time to take minimum over
    nu= 8          # number of subwindows
    qith= np.array([0.03, 0.05, 0.06, np.Inf],dtype=float) # noise slope thresholds in dB/s
    nsmdb= np.array([47, 31.4, 15.7, 4.1],dtype=float) # maximum permitted +ve noise slope in dB/s


    # derived algorithm constants
    aca = np.exp(-hop_length/taca) # smoothing constant for alpha_c in equ (11) = 0.7
    acmax=aca          # min value of alpha_c = 0.7 in equ (11) also = 0.7
    amax=np.exp(-hop_length/tamax) # max smoothing constant in (3) = 0.96
    aminh=np.exp(-hop_length/taminh) # min smoothing constant (upper limit) in (3) = 0.3
    bmax=np.exp(-hop_length/tbmax) # max smoothing constant in (20) = 0.8
    SNRexp = -hop_length/tpfall
    nv=round(td/(hop_length*nu))    # length of each subwindow in frames


    if nv<4:        # algorithm doesn't work for miniscule frames
        nv=4
        nu=round(td/(hop_length*nv))
    nd=nu*nv           # length of total window in frames
    (md,hd,dd) = mhvals(nd) # calculate the constants M(D) and H(D) from Table III
    (mv,hv,dv) = mhvals(nv) # calculate the constants M(D) and H(D) from Table III
    nsms=np.array([10])**(nsmdb*nv*hop_length/10)  # [8 4 2 1.2] in paper
    qeqimax=1/qeqmin  # maximum value of Qeq inverse (23)
    qeqimin=1/qeqmax # minumum value of Qeq per frame inverse
    
    t       = param_em.t 
    p       = param_em.p 
    ac      = param_em.ac               # correction factor (9)
    sn2     = param_em.sn2              # estimated noise power
    pb      = param_em.pb           # smoothed noisy speech power (20)
    pb2     = param_em.pb2
    pminu   = param_em.pminu
    actmin  = param_em.actmin
    actminsub = param_em.actminsub
    if (param_em.subwc == 0):
        subwc = nv
    else:
        subwc   = param_em.subwc                   # force a buffer switch on first loop
    if (type(param_em.actbuf) == int):
        actbuf  = np.array(np.ones((nu,nFFT2)) * np.Inf)
    else:    
        actbuf  = param_em.actbuf
    ibuf    = param_em.ibuf
    lminflag = param_em.lminflag
 
    # loop for each frame
   
    acb=(1+(sum(p) / (sum(pSpectrum) + 1e-9)-1)**2)**(-1)    # alpha_c-bar(t)  (9) ( + 1e-9 add by adelino)
    
    tmp=np.array([acb] )
    tmp[tmp < acmax] = acmax
    #max_complex(np.array([acb] ),np.array([acmax] ))
    
    ac=aca*ac+(1-aca)*tmp    # alpha_c(t)  (10)
    
    ah=amax*ac*(1+(p/sn2-1)**2)**(-1)       # alpha_hat: smoothing factor per frequency (11)
    SNR=sum(p)/sum(sn2)
    
           
    ah=max_complex(ah,min_complex(np.array([aminh] ),np.array([SNR**SNRexp] )))                            # lower limit for alpha_hat (12)

    p=ah*p+(1-ah)*pSpectrum            # smoothed noisy speech power (3)

    b=min_complex(ah**2,np.array([bmax] )) # smoothing constant for estimating periodogram variance (22 + 2 lines)
    pb=b*pb + (1-b)*p            # smoothed periodogram (20)
    pb2=b*pb2 + (1-b)*p**2     	 # smoothed periodogram squared (21)

    qeqi = max_complex(min_complex((pb2-pb**2)/(2*sn2**2),np.array([qeqimax] )),np.array([qeqimin/(t+1)] ))   # Qeq inverse (23)
    qiav=sum(qeqi)/nFFT2              # Average over all frequencies (23+12 lines) (ignore non-duplication of DC and nyquist terms)
    bc=1+av*np.sqrt(qiav)              # bias correction factor (23+11 lines)
    bmind=1+2*(nd-1)*(1-md)/(qeqi**(-1)-2*md)      # we use the signalmplified form (17) instead of (15)
    bminv=1+2*(nv-1)*(1-mv)/(qeqi**(-1)-2*mv)    # same expressignalon but for sub windows
    kmod=(bc*p*bmind) < actmin      # Frequency mask for new minimum

    if any(kmod):
        actmin[kmod]=bc*p[kmod]*bmind[kmod]
        actminsub[kmod]=bc*p[kmod]*bminv[kmod]

    if (subwc > 1) and (subwc < nv):              # middle of buffer - allow a local minimum
        lminflag= np.logical_or(lminflag,kmod)    	# potential local minimum frequency bins
        pminu=min_complex(actminsub,pminu)
        sn2=pminu.copy()
    else:
        if subwc>=nv:                   # end of buffer - do a buffer switch
            ibuf=1+(ibuf%nu)     	# increment actbuf storage pointer
            actbuf[ibuf-1,:]=actmin.copy()   	# save sub-window minimum
            pminu=min_complex_mat(actbuf)
            i=np.nonzero(np.array(qiav )<qith)
            nsm=nsms[i[0][0]]     	# noise slope max
            lmin = np.logical_and(np.logical_and(np.logical_and(lminflag,np.logical_not(kmod)),actminsub<(nsm*pminu)),actminsub>pminu)
            if any(lmin):
                pminu[lmin]=actminsub[lmin]
                actbuf[:,lmin]= np.ones((nu,1)) * pminu[lmin]
            lminflag[:]=0
            actmin[:]=np.Inf
            subwc=0

    subwc=subwc+1
    x = sn2.copy()
    param_em.set_param(t+1,p,ac,sn2,pb,pb2,pminu,actmin, actminsub, subwc, actbuf, ibuf, lminflag)
    return x, param_em

def mhvals(*args):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    nargin = len(args)

    dmh=np.array([
        [1,   0,       0],
        [2,   0.26,    0.15],
        [5,   0.48,    0.48],
        [8,   0.58,    0.78],
        [10,  0.61,    0.98],
        [15,  0.668,   1.55],
        [20,  0.705,   2],
        [30,  0.762,   2.3],
        [40,  0.8,     2.52],
        [60,  0.841,   3.1],
        [80,  0.865,   3.38],
        [120, 0.89,    4.15],
        [140, 0.9,     4.35],
        [160, 0.91,    4.25],
        [180, 0.92,    3.9],
        [220, 0.93,    4.1],
        [260, 0.935,   4.7],
        [300, 0.94,    5]
        ],dtype=float)

    if nargin>=1:
        d=args[0]
        i=np.nonzero(d<=dmh[:,0])
        if len(i)==0:
            i=np.shape(dmh)[0]-1
            j=i
        else:
            i=i[0][0]
            j=i-1
        if d==dmh[i,0]:
            m=dmh[i,1]
            h=dmh[i,2]
        else:
            qj=np.sqrt(dmh[i-1,0])    # interpolate usignalng sqrt(d)
            qi=np.sqrt(dmh[i,0])
            q=np.sqrt(d)
            h=dmh[i,2]+(q-qi)*(dmh[j,2]-dmh[i,2])/(qj-qi)
            m=dmh[i,1]+(qi*qj/q-qj)*(dmh[j,1]-dmh[i,1])/(qi-qj)
    else:
        d=dmh[:,0].copy()
        m=dmh[:,1].copy()
        h=dmh[:,2].copy()

    return m,h,d


def max_complex(a,b):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    if len(a)==1 and len(b)>1:
        a=np.tile(a,np.shape(b))
    if len(b)==1 and len(a)>1:
        b=np.tile(b,np.shape(a))

    i=np.logical_or(np.iscomplex(a),np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i]=np.absolute(aa[i])
        bb[i]=np.absolute(bb[i])
    if a.dtype == 'complex' or b.dtype== 'complex':
        cc = np.array(np.zeros(np.shape(a)) )
    else:
        cc = np.array(np.zeros(np.shape(a)),dtype=float)

    i=aa>bb
    cc[i]=a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc

def min_complex(a,b):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    if len(a)==1 and len(b)>1:
        a=np.tile(a,np.shape(b))
    if len(b)==1 and len(a)>1:
        b=np.tile(b,np.shape(a))

    i=np.logical_or(np.iscomplex(a),np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i]=np.absolute(aa[i])
        bb[i]=np.absolute(bb[i])

    if a.dtype == 'complex' or b.dtype== 'complex':
        cc = np.array(np.zeros(np.shape(a)) )
    else:
        cc = np.array(np.zeros(np.shape(a)),dtype=float)

    i=aa<bb
    cc[i]=a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc

def min_complex_mat(a):
    """
    This is python implementation of [1],[2], and [3]. 
    
    Refs:
       [1] Rainer Martin.
           Noise power spectral density estimation based on optimal smoothing and minimum statistics.
           IEEE Trans. Speech and Audio Processing, 9(5):504-512, July 2001.
       [2] Rainer Martin.
           Bias compensation methods for minimum statistics noise power spectral density estimation
           Signal Processing, 2006, 86, 1215-1229
       [3] Dirk Mauler and Rainer Martin
           Noise power spectral density estimation on highly correlated data
           Proc IWAENC, 2006
    
         Copyright (C) Mike Brookes 2008
         Version: $Id: estnoisem.m 1718 2012-03-31 16:40:41Z dmb $
    
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    s=np.shape(a)
    m = np.array(np.zeros(s[1]) )
    for i in range(0,s[1]):
        j = np.argmin(np.absolute(a[:,i]))
        m[i] = a[j,i]
    return m
