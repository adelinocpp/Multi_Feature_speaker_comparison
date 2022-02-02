#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:12:34 2022

@author: adelino
"""
import numpy as np
import librosa
import scipy
import os
from pathlib import Path
import config as c
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy.matlib as npm

from scipy import signal

# TODO: criar nova biblioteca de funções
# -----------------------------------------------------------------------------
def levinson_aps(r, p):
    X = np.zeros((p,p))
    r_w = r[0:p]
    r_c = r_w[::-1]
    for i in range(0,p):
        if (i == 0):
            X[i,:] = r_w
        else:
            X[i,:] = np.roll(r_c,i+1)
    b = -r[1:p+1]
    a = np.linalg.lstsq(X, b.T,rcond=None)[0]
    G = r[0] - np.matmul(a.T,r[1:p+1])
    a = np.concatenate(([1],a))
    return a, G
# -----------------------------------------------------------------------------
def computeD(x,p=2):
    nDim, nframe = x.shape
    dMtx = np.zeros((nDim, nframe))
    for i in range(p,nframe- p):
        n = np.zeros((nDim,))
        d = 0;
        for j in range(1,p+1):
            n0 = j*(x[:,i+j] - x[:,i-j])
            n0.shape = (nDim,1)
            n = np.add(n,j*(x[:,i+j] - x[:,i-j]))
            d += j^2;
    
        dMtx[:,i] = np.divide(n,(2*d))

    return dMtx
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
        # a, e = solve_lpc(x[:,idx], order)
        a, e = levinson_aps(x[:,idx], order)
        y[:,idx] = a/e
        # y[:,idx] = lpc2cep(a/e,order+1)
    return y
# -----------------------------------------------------------------------------
def rasta_filter(spec):
    numer = np.array(range(-2,3))
    numer = -numer /np.sum(numer*numer);
    denom = np.array([1 -0.94]);
    zi = signal.lfilter_zi(numer, [1])
    y, z = signal.lfilter(numer, [1], spec[:4],axis=0,zi=0*zi)
    y0 = y*0
    y1 = signal.lfilter(numer, denom, spec[4:],axis=0,zi=z)
    return np.concatenate((y0,y1[0]),axis=0)
# -----------------------------------------------------------------------------
def bark2hertz(bark_freq):
    return 600 * np.sinh(bark_freq/6)
# -----------------------------------------------------------------------------
def hertz2bark(hertz_freq):
    return 6*np.arcsinh(hertz_freq/600)
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
def build_folders_to_save(file_name):
    split_path = file_name.split('/')
    for idx in range(1,len(split_path)):
        curDir = '/'.join(split_path[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
# -----------------------------------------------------------------------------
class Feature:
    def __init__(self,data=np.empty([]),computed=True):
        self.data = data
        self.computed = computed
# -----------------------------------------------------------------------------
class AcousticsFeatures:
    def __init__(self, file_name= '', win_length=0.025, step_length = 0.01):
        self.win_length = win_length
        self.step_length = step_length
        self.computed = False
        self.file_name = file_name
        self.feature_file = ''
        self.sample_rate = 0;
        self.features = {"spectogram":  Feature(),
                         "time_domain": np.empty([]),
                         "freq_domain": np.empty([]),
                         "spc_entropy": Feature(),
                         "LTAS":        Feature(),
                         "vad_shon":    Feature(),
                         "S2NR":        Feature(),
                         "pitch":       Feature(),
                         "formants":    Feature(),
                         "mfcc":        Feature(),
                         "pncc":        Feature(),
                         "plp":         Feature(computed=False),
                         "rasta_plp":   Feature(),
                         "ssch":        Feature(),
                         "zcpa":        Feature(),
                         "teocc":       Feature(),
                         "mfec":        Feature()
                         };
    
    def check_compute_completed(self, feature_name = ''):
        return_value = True
        if (feature_name == ''):
            for idx in enumerate(self.features):
                key = list(self.features)[idx]
                return_value = return_value and self.features[key].computed
            return return_value
        else:
            if not (feature_name in self.features):
                return False
            else:
                return self.features[feature_name].computed;
        return False
    
    def save_preps(self, audio_path, feature_path):
        file_stem = Path(self.file_name).stem
        audio_file = Path(self.file_name).name
        file_feature = self.file_name.replace(audio_path,feature_path)
        file_feature = file_feature.replace(audio_file,file_stem + '.p')
        build_folders_to_save(file_feature)        
        self.feature_file = file_feature
        
    def get_feature_file(self):
        return self.feature_file
# -----------------------------------------------------------------------------
    def compute_features(self):
        if (self.file_name == ''):
            return
        audio, sr = librosa.load(self.file_name, sr=None, mono=True)
        self.sample_rate = sr
        n_win_length = int(self.sample_rate*self.win_length)
        n_step_length = int(self.sample_rate*self.step_length)
        # h_win_length = int(self.sample_rate*self.win_length)
        num_samples = len(audio)
        n_FFT = int(2 ** np.ceil(np.log2(n_win_length)))
        h_FFT = int(0.5*n_FFT)
        # --- algumas janelas espectrais
        hamming_win = np.hamming(n_win_length)
        hann_win = np.hanning(n_win_length)
        
        data_spectogram = np.empty((h_FFT,0))
        data_freq_domain = np.array([0.5*i*self.sample_rate/h_FFT for i in range(0,h_FFT)])
        data_time = np.empty((1,0));
        
        # --- Filtros para componentes
        plp_bark_filters = build_bark_filters(n_FFT,self.sample_rate,nfilts=c.NUM_PLP_FILTERS)
        
        # --- Variaveis auxiliares
                
        aud_spec = np.empty((c.NUM_PLP_FILTERS,0))
        # -- - INICIO DO CALCULO POR FRAME ------------------------------------
        for time_idx in range(0,num_samples-n_win_length+1,n_step_length):
            win_audio = audio[time_idx:time_idx+n_win_length]
            data_time = np.append(data_time,(time_idx+0.5*n_win_length)/self.sample_rate);
            
            # -- ESPECTOGRAMA, MFCC
            if (not self.features["spectogram"].computed):
                win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
                abs_fft = np.abs(win_fft[:h_FFT])
                # ---
                spec = 20*np.log10(abs_fft)
                spec.shape = (h_FFT,1)
                data_spectogram = np.append(data_spectogram,spec,axis=1)
            
            
            # -- PLP, RASTA PLP -----------------------------------------------
            if (not self.features["plp"].computed) or (not self.features["rasta_plp"].computed):
                win_fft = scipy.fft.fft(win_audio*(32*1024)*hann_win,n=n_FFT)
                frame_aud_spec = np.empty((c.NUM_PLP_FILTERS,1))
                abs_fft = np.power(np.abs(win_fft[:h_FFT]),2)
                for idx_aud in range(0,c.NUM_PLP_FILTERS):    
                    if (c.PLP_SUM_POWER):
                        frame_aud_spec[idx_aud] = np.matmul(plp_bark_filters[idx_aud,:],abs_fft)    
                    else:
                        frame_aud_spec[idx_aud] = np.power(\
                                    np.matmul(plp_bark_filters[idx_aud,:],
                                    np.sqrt(abs_fft)),2)
                aud_spec = np.append(aud_spec,frame_aud_spec,axis=1)
            
  
            # TODO: implementar depois de calcular as bandas

        # --- FIM DO CALCULO POR FRAME -----------------------------------------       
        # --- Entropia espectral -----------------------------------------------
        if (not self.features["spc_entropy"].computed):
            prob_mtx = np.empty(data_spectogram.shape)
            for idx in range (0,h_FFT):
                X = data_spectogram[idx,:][:, np.newaxis]    
                kde = KernelDensity(kernel='gaussian').fit(X)
                prob_mtx[idx,:] = np.exp(kde.score_samples(X))
                entropy_mtx = np.sum(np.multiply(prob_mtx,np.log(prob_mtx)),axis=0)        
        # ----------------------------------------------------------------------
        # CONTINUA PLP e RASTA-PLP
        if (not self.features["plp"].computed):
            aspectrum_plp = postaud(aud_spec,sr/2)
            
            lpc = do_lpc(aspectrum_plp,c.NUM_CEPS_COEFS)
            cep = lpc2cep(lpc,c.NUM_CEPS_COEFS+1)
            # TODO: verificar dimensoes PLP e conferir com matlab
            plp_mtx   = lifter(cep,0.6)
            d_plp_mtx = computeD(plp_mtx)
            dd_plp_mtx = computeD(d_plp_mtx)
            plp_mtx = np.concatenate((plp_mtx,d_plp_mtx,dd_plp_mtx),axis=0)
        if (not self.features["rasta_plp"].computed):
            aspectrum_rasta = np.empty(aud_spec.shape)
            for idx in range(0,aud_spec.shape[0]):
                aspectrum_rasta[idx,:] = np.exp(rasta_filter(np.log(aud_spec[idx,:])))
            # aspectrum_rasta = lifter(do_lpc2cep(aspectrum_rasta,c.NUM_CEPS_COEFS),0.6)
            
         #lpcas = dolpc(postspectrum, modelorder);
         # convert lpc to cepstra
         #cepstra = lpc2cep(lpcas, modelorder+1);
         # .. or to spectra
         #[spectra,F,M] = lpc2spec(lpcas, nbands);
            
        # x_vals = np.linspace(data_spectogram[0,:].min(),data_spectogram[0,:].max(),200)
        # plt.plot(x_vals,density(x_vals))
        # plt.show()

        
        self.features["time_domain"] = data_time
        self.features["freq_domain"] = data_freq_domain
        if (not self.features["spectogram"].computed):
            self.features["spectogram"].data = data_spectogram
            self.features["spectogram"].computed = True    
        if (not self.features["spc_entropy"].computed):
            self.features["spc_entropy"].data = entropy_mtx
            self.features["spc_entropy"].computed = True        
        if (not self.features["plp"].computed):
            self.features["plp"].data = plp_mtx
            self.features["plp"].computed = True        
        if (not self.features["rasta_plp"].computed):
            self.features["rasta_plp"].data = aspectrum_rasta
            self.features["rasta_plp"].computed = True        
            
            
            
            
            
            
            
