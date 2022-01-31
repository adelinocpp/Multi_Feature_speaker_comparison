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

from scipy import signal
# -----------------------------------------------------------------------------
def computeD(x,p=2):
    return x
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
def make_matrix_X(x, p):
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])
    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i
        X[i, :] = xz[offset : offset + p]
    return X
# -----------------------------------------------------------------------------
def solve_lpc(x, p):
    b = x[1:]
    X = make_matrix_X(x, p)
    a = np.linalg.lstsq(X, b.T,rcond=None)[0]
    e = b - np.dot(X, a)
    g = np.var(e)
    a = np.concatenate(([1],a))
    return [a, g]
# -----------------------------------------------------------------------------
def lpc2cep(a,nout=0):
    nin = a.shape[0]
    order = nin - 1;
    if (nout == 0):
        nout = order + 1;
    c = np.zeros(nout,)
    c[0] = -np.log(a[0])
    
    a = a/a[0]
    for n in range(1,nout):
        Soma = 0
        for m in range(1,n):
            Soma += (n - m) * a[m] * c[n - m]
            
        c[n] = -(a[n] + Soma/n);
    return c
# -----------------------------------------------------------------------------
def do_lpc2cep(spec, order=8):
    [nbands, nframes] = spec.shape
    x = np.concatenate((spec,np.flipud(spec)),axis=1)
    x = np.real(np.fft.ifft(x,axis=1))
    x = x[:nbands,:]
    y = np.zeros((order+1,nframes))
    for idx in range(0,nframes):
        a, e = solve_lpc(x[:,idx], order)
        # y[:,idx] = a/e
        y[:,idx] = lpc2cep(a/e,order+1)
    # y = np.divide()
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
def hertz2bark(hertz_freq):
    # Converts frequencies Hertz (Hz) to Bark
    return 6*np.arcsinh(hertz_freq/600)
# -----------------------------------------------------------------------------
def build_bark_filters(nfft,sr,nfilts=0,width=1,min_freq=0,max_freq=0):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_bark = hertz2bark(min_freq)
    nyq_bark = hertz2bark(max_freq) - min_bark
    if (nfilts == 0):
        nfilts = np.ceil(nyq_bark)+1
    h_fft = int(0.5*nfft)
    wts = np.zeros((nfilts, h_fft));

    step_barks = nyq_bark/(nfilts-1);
    bin_barks = np.array([hertz2bark(0.5*i*sr/nfft) for i in range(0,h_fft)])
    limits = np.empty((2,h_fft))
    for i in range(0,nfilts):
        f_bark_mid = min_bark + i*step_barks;
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        limits[0,:] = -2.5*(bin_barks - f_bark_mid - 0.5)
        limits[1,:] = (bin_barks - f_bark_mid + 0.5)
        # lof = (bin_barks - f_bark_mid - 0.5);
        # hif = (bin_barks - f_bark_mid + 0.5);
        # wts[i,:] = np.power(10, np.min(0, np.min([hif; -2.5*lof])/width))
        wts[i,:] = np.power(10, np.minimum(np.zeros((h_fft,)), np.min(limits,axis=0)/width))
    return wts[:,:int(0.5*nfft)]
    
# -----------------------------------------------------------------------------
def build_folders_to_save(file_name):
    split_path = file_name.split('/')
    for idx in range(1,len(split_path)):
        curDir = '/'.join(split_path[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
# -----------------------------------------------------------------------------
class Feature:
    def __init__(self):
        self.data = np.empty([])
        self.computed = False
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
                         "plp":         Feature(),
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
        # -- - INICIO DO CALCULO POR FRAME -------------------------------------
        for time_idx in range(0,num_samples-n_win_length+1,n_step_length):
            win_audio = audio[time_idx:time_idx+n_win_length]
            data_time = np.append(data_time,(time_idx+0.5*n_win_length)/self.sample_rate);
            # -- ESPECTOGRAMA, MFCC
            win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
            abs_fft = np.abs(win_fft[:h_FFT])
            # ---
            spec = 20*np.log10(abs_fft)
            spec.shape = (h_FFT,1)
            data_spectogram = np.append(data_spectogram,spec,axis=1)
            
            # -- PLP, RASTA PLP
            win_fft = scipy.fft.fft(win_audio*(32*1024)*hann_win,n=n_FFT)
            frame_aud_spec = np.empty((c.NUM_PLP_FILTERS,1))
            nl_aspectrum = np.empty((c.NUM_PLP_FILTERS,0))
            for idx_aud in range(0,c.NUM_PLP_FILTERS):
                if (c.PLP_SUM_POWER):
                    abs_fft = np.power(np.abs(win_fft[:h_FFT]),2)
                    frame_aud_spec[idx_aud] = np.power(\
                        np.matmul(plp_bark_filters[idx_aud,:],\
                                    np.power(np.abs(win_fft[:h_FFT]),2) ),2)
                else:
                    frame_aud_spec[idx_aud] = np.power(\
                        np.matmul(plp_bark_filters[idx_aud,:],\
                                    np.abs(win_fft[:h_FFT])),2)
            # PLP e RASTA-PLP       
            aud_spec = np.append(aud_spec,frame_aud_spec,axis=1)
            
  
            # TODO: implementar depois de calcular as bandas

        # --- FIM DO CALCULO POR FRAME -----------------------------------------       
        # --- Entropia espectral -----------------------------------------------
        prob_mtx = np.empty(data_spectogram.shape)
        for idx in range (0,h_FFT):
            X = data_spectogram[idx,:][:, np.newaxis]    
            kde = KernelDensity(kernel='gaussian').fit(X)
            prob_mtx[idx,:] = np.exp(kde.score_samples(X))
        entropy_mtx = np.sum(np.multiply(prob_mtx,np.log(prob_mtx)),axis=0)        
        # ----------------------------------------------------------------------
        # CONTINUA PLP e RASTA-PLP
        aspectrum_rasta = np.empty(aud_spec.shape)
        aspectrum_plp = np.log(aud_spec)
        for idx in range(0,aud_spec.shape[0]):
            aspectrum_rasta[idx,:] = np.exp(rasta_filter(np.log(aud_spec[idx,:])))
        
        aspectrum_rasta = lifter(do_lpc2cep(aspectrum_rasta,c.NUM_CEPS_COEFS),0.6)
        aspectrum_plp   = lifter(do_lpc2cep(aspectrum_plp,c.NUM_CEPS_COEFS),0.6)
    
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
        self.features["spectogram"].data = data_spectogram
        self.features["spectogram"].computed = True    
        self.features["spc_entropy"].data = entropy_mtx
        self.features["spc_entropy"].computed = True        
            
            
            
            
            
            
            
            
