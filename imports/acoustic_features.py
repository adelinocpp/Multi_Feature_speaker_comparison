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
# import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from .build_filters import build_bark_filters
from .rasta_plp_functions import postaud, do_lpc, lpc2cep, lifter, rasta_filter
from scipy import signal

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
            d += j**2;
        dMtx[:,i] = np.divide(n,(2*d))
    return dMtx    
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
                         "vad_shon":    Feature(computed=False),
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
        # --- pre-enfasis ------------------------------------------------------
        audio = signal.lfilter(np.array([1, -0.975]), np.array([1]), audio,axis=0)
        
        n_win_length = int(self.sample_rate*self.win_length)
        n_step_length = int(self.sample_rate*self.step_length)
        
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
            if (not self.features["spectogram"].computed) or (not self.features["mfcc"].computed):
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
            plp_mtx   = lifter(cep,0.6)
            d_plp_mtx = computeD(plp_mtx)
            dd_plp_mtx = computeD(d_plp_mtx)
            plp_mtx = np.concatenate((plp_mtx,d_plp_mtx,dd_plp_mtx),axis=0)
        if (not self.features["rasta_plp"].computed):
            aspectrum_rasta = np.empty(aud_spec.shape)
            for idx in range(0,aud_spec.shape[0]):
                aspectrum_rasta[idx,:] = np.exp(rasta_filter(np.log(aud_spec[idx,:])))         
            aspectrum_rasta = postaud(aspectrum_rasta,sr/2)            
            lpc = do_lpc(aspectrum_rasta,c.NUM_CEPS_COEFS)
            cep = lpc2cep(lpc,c.NUM_CEPS_COEFS+1)
            rasta_plp_mtx   = lifter(cep,0.6)
            d_rasta_plp_mtx = computeD(rasta_plp_mtx)
            dd_rasta_plp_mtx = computeD(d_rasta_plp_mtx)
            rasta_plp_mtx = np.concatenate((rasta_plp_mtx,d_rasta_plp_mtx,dd_rasta_plp_mtx),axis=0)
        
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
            self.features["rasta_plp"].data = rasta_plp_mtx
            self.features["rasta_plp"].computed = True        
            
            
            
            
            
            
            
