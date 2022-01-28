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

# ------------------------------------------------------------------------------
def rasta_filter(spec):
    # TODO: implementar
    return np.zeros(spec.shape)

# ------------------------------------------------------------------------------
def hertz2bark(hertz_freq):
    # Converts frequencies Hertz (Hz) to Bark
    return 6*np.arcsinh(hertz_freq/600)
# ------------------------------------------------------------------------------
def build_bark_filters(nfft,sr,nfilts=0,width=1,min_freq=0,max_freq=0):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_bark = hertz2bark(min_freq)
    nyq_bark = hertz2bark(max_freq) - min_bark
    if (nfilts == 0):
        nfilts = np.ceil(nyq_bark)+1
        
    wts = np.zeros((nfilts, nfft));

    step_barks = nyq_bark/(nfilts-1);
    bin_barks = np.array([hertz2bark(0.5*i*sr/nfft) for i in range(0,nfft)])
    limits = np.empty((2,nfft))
    for i in range(0,nfilts):
        f_bark_mid = min_bark + i*step_barks;
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        limits[0,:] = -2.5*(bin_barks - f_bark_mid - 0.5)
        limits[1,:] = (bin_barks - f_bark_mid + 0.5)
        # lof = (bin_barks - f_bark_mid - 0.5);
        # hif = (bin_barks - f_bark_mid + 0.5);
        # wts[i,:] = np.power(10, np.min(0, np.min([hif; -2.5*lof])/width))
        wts[i,:] = np.power(10, np.minimum(np.zeros((nfft,)), np.min(limits,axis=0)/width))
    return wts[:,:int(0.5*nfft)]
    
# ------------------------------------------------------------------------------
def build_folders_to_save(file_name):
    split_path = file_name.split('/')
    for idx in range(1,len(split_path)):
        curDir = '/'.join(split_path[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
# ------------------------------------------------------------------------------
class Feature:
    def __init__(self):
        self.data = np.empty([])
        self.computed = False
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
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
                
        aud_spec = np.empty(c.NUM_PLP_FILTERS,0)
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
            frame_aud_spec = np.empty((c.NUM_PLP_FILTER,))
            for idx_aud in range(0,c.NUM_PLP_FILTERS):
                if (c.PLP_SUM_POWER):
                    abs_fft = np.power(np.abs(win_fft[:h_FFT]),2)
                    frame_aud_spec[idx_aud] = np.power(\
                        np.multiply(plp_bark_filters[idx_aud,:],\
                                    np.power(np.abs(win_fft[:h_FFT]),2) ),2)
                else:
                    frame_aud_spec[idx_aud] = np.power(\
                        np.multiply(plp_bark_filters[idx_aud,:],\
                                    np.abs(win_fft[:h_FFT])),2)
                    
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
        
        # x_vals = np.linspace(data_spectogram[0,:].min(),data_spectogram[0,:].max(),200)
        # plt.plot(x_vals,density(x_vals))
        # plt.show()

        
        self.features["time_domain"] = data_time
        self.features["freq_domain"] = data_freq_domain
        self.features["spectogram"].data = data_spectogram
        self.features["spectogram"].computed = True    
        self.features["spc_entropy"].data = entropy_mtx
        self.features["spc_entropy"].computed = True        
            
            
            
            
            
            
            
            