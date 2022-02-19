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
from .build_filters import build_bark_plp_filters, build_mel_tiang_filters, \
                        build_erb_gamma_filters
from .rasta_plp_functions import postaud, do_lpc, lpc2cep, lifter, rasta_filter
from scipy import signal
from .vad_functions import estnoisem_frame, bessel
from .pncc_functions import queue
import math

# -----------------------------------------------------------------------------
def dct_aps(x, K=0):
    N = x.shape[0]
    if (K == 0) or (K > N):
        K = N
    c = np.zeros((K,1))
    Mconst = np.sqrt(2/K)
    for k in range(0,K):
        if (k == 0):
            beta = 1/np.sqrt(2)
        else:
            beta = 1
        kSum = [x[n]*np.cos(k*np.pi/N*(0.5+n)) for n in range(0,N)]
        c[k] = Mconst*beta*np.sum(kSum)
    return c
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
        self.features = {"spectogram":  Feature(), # v
                         "time_domain": np.empty([]), # v
                         "freq_domain": np.empty([]), # v
                         "spc_entropy": Feature(), # v
                         "LTAS":        Feature(),
                         "vad_sonh":    Feature(computed = False), #v
                         "S2NR":        Feature(),
                         "pitch":       Feature(),
                         "formants":    Feature(),
                         "mfcc":        Feature(), # v
                         "pncc":        Feature(), # v
                         "plp":         Feature(), # v
                         "rasta_plp":   Feature(), # v
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
        # time_s = np.array([n/sr for n in range (0,len(audio))])
        self.sample_rate = sr
        # ======================================================================
        # --- pre-enfasis ------------------------------------------------------
        audio = signal.lfilter(np.array([1, -0.975]), np.array([1]), audio,axis=0)
        # ======================================================================
        num_samples = len(audio)        
        n_win_length = int(np.ceil(self.sample_rate*self.win_length))
        n_step_length = int(np.ceil(self.sample_rate*self.step_length))
        # ======================================================================
        if (np.mod((num_samples - n_win_length),n_step_length) == 0):
            self.n_frames = int(np.floor((num_samples - n_win_length) / n_step_length))
        else:
            self.n_frames = int(np.floor((num_samples - n_win_length) / n_step_length) + 1)
        # ======================================================================
        # --- NFFT base
        n_FFT = int(2 ** np.ceil(np.log2(n_win_length)))
        h_FFT = int(0.5*n_FFT)
        # ======================================================================
        # --- algumas janelas espectrais
        hamming_win = np.hamming(n_win_length)
        hann_win = np.hanning(n_win_length)
        
        
        # ======================================================================
        # --- Variaveis auxiliares
        # --- PNCC
        wFreq      = 40 # Changed to 40 from 200 as low freq is 40 in gabor as well
        p_FFT = 1024
        q_FFT = int(0.5*p_FFT)
        dLamda_L = 0.999
        dLamda_S = 0.999
        dPowerCoeff = 1/15
        dFactor = 2.0
        iM = 0 # Changed from 2 to 0 as number of frames coming out to be different due to queue
        iN = 4
        iSMType = 0;
        dLamda  = 0.999
        dLamda2 = 0.5
        dDelta1 = 1
        dLamda3 = 0.85
        dDelta2 = 0.2
        bSSF        = True
        bPowerLaw   = True
        MEAN_POWER = 1e10
        dMean  = 4e+07 #4.0638e+07 #  --> 4.0638e+07 from WSJ0-si84 -- 5.8471e+08 from code
        i_FI_Out = 0
        # --- VAD 
        maxPosteriorSNR= 1000   
        minPosteriorSNR= 0.0001    
        theshold = 0.7
       
        a01=self.step_length/0.05     # a01=P(signallence->speech)  hop_length/mean signallence length (50 ms)
        a00=1-a01               # a00=P(signallence->signallence)
        a10=self.step_length/0.1      # a10=P(speech->signallence) hop/mean talkspurt length (100 ms)
        a11=1-a10               # a11=P(speech->speech)
    
        b01=a01/a00
        b10=a11-a10*a01/a00
      
        smoothFactorDD=0.99
        previousGainedaPosSNR=1 
        # (nFrames,nFFT2) = pSpectrum.shape                
        probRatio=np.zeros((self.n_frames,1))
        logGamma_frame=0           
        # ======================================================================
        # --- Filtros para componentes
        plp_bark_filters = build_bark_plp_filters(n_FFT,self.sample_rate,\
                    nfilts=c.NUM_PLP_FILTERS)
        mfcc_mel_filters = build_mel_tiang_filters(n_FFT,self.sample_rate,\
                    nfilts=c.NUM_MFCC_FILTERS)
        gamma_mel_filters = build_erb_gamma_filters(p_FFT,self.sample_rate,\
                    nfilts=c.NUM_PNCC_FILTERS,min_freq=wFreq)            
        teocc_mel_filters = build_erb_gamma_filters(p_FFT,self.sample_rate,\
                    nfilts=c.NUM_PNCC_FILTERS,min_freq=wFreq) 
        # ======================================================================
        # --- VETORES E MATRIZES DE SAIDA --------------------------------------
        data_spectogram = np.empty((h_FFT,0))
        data_freq_domain = np.array([0.5*i*self.sample_rate/h_FFT for i in range(0,h_FFT)])
        data_time = np.empty((1,0))
        # --- VAD 
        vad_sonh = np.empty((1,0))
        # --- MFEC
        mfec_spec = np.empty((c.NUM_CEPS_COEFS,0))
        # --- TEOCC
        teocc_spec = np.empty((c.NUM_CEPS_COEFS,0))
        # --- PLP e RAST-PLP         
        aud_spec = np.empty((c.NUM_PLP_FILTERS,0))
        # --- MFCC
        mfcc_spec = np.empty((c.NUM_CEPS_COEFS,0))
        # --- PNCC
        pncc_spec = np.empty((c.NUM_CEPS_COEFS,0))
        adSumPower = np.zeros((self.n_frames,))
        # ======================================================================
        # -- - INICIO DO CALCULO POR FRAME ------------------------------------
        t_frame = 0
        for time_idx in range(0,num_samples-n_win_length+1,n_step_length):
            win_audio = audio[time_idx:time_idx+n_win_length]
            data_time = np.append(data_time,(time_idx+0.5*n_win_length)/self.sample_rate);
            
            # -- ESPECTOGRAMA, MFCC
            if (not self.features["spectogram"].computed):
                # --- ESPECTOGRAMA
                win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
                abs_fft = np.abs(win_fft[:h_FFT])
                spec = 20*np.log10(abs_fft)
                spec.shape = (h_FFT,1)
                data_spectogram = np.append(data_spectogram,spec,axis=1)
            # --- MFCC
            if (not self.features["mfcc"].computed):
                if not ("abs_fft" in locals()):
                    win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
                    abs_fft = np.abs(win_fft[:h_FFT])
                frame_mag_spec = np.empty((c.NUM_MFCC_FILTERS,1))
                for idx_aud in range(0,c.NUM_MFCC_FILTERS): 
                    frame_mag_spec[idx_aud] = np.matmul(mfcc_mel_filters[idx_aud,:],abs_fft)
                frame_cep = dct_aps(np.log(frame_mag_spec),c.NUM_CEPS_COEFS)
                mfcc_spec = np.append(mfcc_spec,frame_cep,axis=1)
                
            
            # --- VAD
            if (not self.features["vad_sonh"].computed):
                if not ("abs_fft" in locals()):
                    win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
                    abs_fft = np.abs(win_fft[:h_FFT])
                pwr_fft = np.power(abs_fft,2)
                if (t_frame == 0):
                    noise_pwd = pwr_fft
                    
                estNoise = estnoisem_frame(pwr_fft,self.step_length, noise_pwd)   
                aPosterioriSNR_frame = np.divide(pwr_fft,estNoise)
                aPosterioriSNR_frame[aPosterioriSNR_frame > maxPosteriorSNR] = maxPosteriorSNR
                aPosterioriSNR_frame[aPosterioriSNR_frame < minPosteriorSNR] = minPosteriorSNR
                #operator [2](52)
                oper=aPosterioriSNR_frame-1
                oper[oper < 0] = 0 
                smoothed_a_priori_SNR = smoothFactorDD * previousGainedaPosSNR + (1-smoothFactorDD) * oper
                
                #V for MMSE estimate ([2](8)) 
                V=0.1*smoothed_a_priori_SNR*aPosterioriSNR_frame/(1+smoothed_a_priori_SNR)            
                
                #geometric mean of log likelihood ratios for individual frequency band  [1](4)
                logLRforFreqBins=2*V-np.log(smoothed_a_priori_SNR+1)              
                # logLRforFreqBins=np.exp(smoothed_a_priori_SNR*aPosterioriSNR_frame/(1+smoothed_a_priori_SNR))/(1+smoothed_a_priori_SNR)
                gMeanLogLRT=np.mean(logLRforFreqBins)       
                logGamma_frame=np.log(a10/a01) + gMeanLogLRT + np.log(b01+b10/( a10+a00*np.exp(-logGamma_frame) ) )
                probRatio = 1/(1+np.exp(-logGamma_frame))
                
                #Calculate Gain function which results from the MMSE [2](7).
                gain = (math.gamma(1.5) * np.sqrt(V)) / aPosterioriSNR_frame * np.exp(-1 * V / 2) * ((1 + V) * bessel(0, V / 2) + V * bessel(1, V / 2))
            
                previousGainedaPosSNR = (gain**2) * aPosterioriSNR_frame
                if (probRatio > theshold):
                    probRatio = 1
                else:
                    probRatio = 0        
                
                vad_sonh = np.append(vad_sonh,probRatio)
             # --- TEOCC
            if (not self.features["teocc"].computed):
                if not ("abs_fft" in locals()):
                    win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
                    abs_fft = np.abs(win_fft[:h_FFT])
                
             # --- MFEC
            if (not self.features["mfec"].computed):
                if not ("abs_fft" in locals()):
                    win_fft = scipy.fft.fft(win_audio*hamming_win,n=n_FFT)
                    abs_fft = np.abs(win_fft[:h_FFT])
                
            # --- PNCC
            if (not self.features["pncc"].computed):
                iNumFilts = c.NUM_PNCC_FILTERS
                # --- outro valor de NFFT
                win_fft = scipy.fft.fft(win_audio*hamming_win,n=p_FFT)
                abs_fft = np.abs(win_fft[:q_FFT])
                frame_mag_spec = np.empty((c.NUM_PNCC_FILTERS,1))
                for idx_aud in range(0,c.NUM_PNCC_FILTERS): 
                    frame_mag_spec[idx_aud] = np.sum(np.power(np.multiply(gamma_mel_filters[idx_aud,:],abs_fft),2))
                if (bSSF):
                    # print('Nothing for while')
                    qObj = queue(iM, iNumFilts)
                    # Ring buffer (using a Queue)
                    if (t_frame > 2 * iM):
                        qObj.queue_poll()
            
                    qObj.queue_offer(frame_mag_spec)
                    ad_Q = qObj.queue_avg(iNumFilts);            
                    if (t_frame == 2*iM):
                        ad_PBias  =  ad_Q* 0.9                        
                    if (t_frame >= 2 * iM):
                        # Bias Update
                        for i in range(0,iNumFilts):
                            if (ad_Q[i] > ad_PBias[i]):
                                ad_PBias[i] = dLamda * ad_PBias[i]  + (1 - dLamda) * ad_Q[i]
                            else:
                                ad_PBias[i] = dLamda2 * ad_PBias[i] + (1 - dLamda2) * ad_Q[i]                
                        ad_Q_Out = np.zeros((iNumFilts,))
                        ad_QMVAvg2 = np.zeros((iNumFilts,))
                        ad_QMVAvg3 = np.zeros((iNumFilts,))
                        ad_QMVPeak = np.zeros((iNumFilts,))
                        for i in range(0,iNumFilts):
                            ad_Q_Out[i] =   np.max([ad_Q[i] - ad_PBias[i], 0])
                            if (t_frame == 2 * iM):
                                ad_QMVAvg2[i]  =  0.9 * ad_Q_Out[i]
                                ad_QMVAvg3[i]  =  ad_Q_Out[i]
                                ad_QMVPeak[i]  =  ad_Q_Out[i]                        
                            if (ad_Q_Out[i] > ad_QMVAvg2[i]):
                                ad_QMVAvg2[i] = dLamda * ad_QMVAvg2[i]  + (1 -  dLamda)  *  ad_Q_Out[i]
                            else:
                                ad_QMVAvg2[i] = dLamda2 * ad_QMVAvg2[i] + (1 -  dLamda2) *  ad_Q_Out[i]
                            dOrg =  ad_Q_Out[i]
                            ad_QMVAvg3[i] = dLamda3 * ad_QMVAvg3[i]
                    
                            if (ad_Q[i] <  dFactor * ad_PBias[i]):
                                ad_Q_Out[i] = ad_QMVAvg2[i]
                            else:
                                if (ad_Q_Out[i] <= dDelta1 *  ad_QMVAvg3[i]):
                                    ad_Q_Out[i] = dDelta2 * ad_QMVAvg3[i]
                            
                            ad_QMVAvg3[i] = np.max([ad_QMVAvg3[i],   dOrg])
                            ad_Q_Out[i] =  np.max([ad_Q_Out[i], ad_QMVAvg2[i]])
                            
                        ad_w = np.divide(ad_Q_Out,np.maximum(ad_Q, np.finfo(float).eps))
                        ad_w_sm = np.zeros((iNumFilts,))
                        for i in range(0,iNumFilts):
                            idx_serie = np.array([int(i) for i in range(np.max([i-iN, 0]),np.min([i+iN,iNumFilts-1]))])
                            if (iSMType == 0):
                                ad_w_sm[i] = np.mean(ad_w[idx_serie])
                            elif (iSMType == 1):
                                ad_w_sm[i] = np.exp(np.mean(np.log(ad_w[idx_serie])))
                            elif (iSMType == 2):
                                ad_w_sm[i] = np.mean(np.power(ad_w[idx_serie],(1/15)))**15
                            elif (iSMType == 3):
                                ad_w_sm[i] = np.mean(np.power(ad_w[idx_serie],15 ))**(1 / 15)
                
                        frame_mag_spec = np.multiply(ad_w_sm, frame_mag_spec)
                        adSumPower[t_frame]   = np.sum(frame_mag_spec)
                
                        if  (adSumPower[i_FI_Out] > dMean):
                            dMean = dLamda_S * dMean + (1 - dLamda_S) * adSumPower[i_FI_Out]
                        else:
                            dMean = dLamda_L * dMean + (1 - dLamda_L) * adSumPower[i_FI_Out]
                
                        frame_mag_spec = (frame_mag_spec/dMean)* MEAN_POWER
                        i_FI_Out += 1                
                else: # bSSF = False
                    adSumPower[t_frame] = np.sum(frame_mag_spec)
                    if (adSumPower[i_FI_Out] > dMean):
                        dMean = dLamda_S * dMean + (1 - dLamda_S) * adSumPower[i_FI_Out]
                    else:
                        dMean = dLamda_S * dMean + (1 - dLamda_L) * adSumPower[i_FI_Out]
                    frame_mag_spec = (frame_mag_spec/dMean)* MEAN_POWER      
                    
                if (bPowerLaw):
                    frame_mag_spec = np.power(frame_mag_spec,dPowerCoeff)
                else:
                    frame_mag_spec = np.log(frame_mag_spec + np.finfo(float).eps)
                pncc_spec = np.append(pncc_spec,dct_aps(frame_mag_spec,c.NUM_CEPS_COEFS),axis=1)
                    
            # -- PLP, RASTA PLP -----------------------------------------------
            if (not self.features["plp"].computed) or (not self.features["rasta_plp"].computed):
                # --- outra janela 
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
            
  
    
            t_frame += 1  
        # --- FIM DO CALCULO POR FRAME -----------------------------------------       
        # ======================================================================
        # TODO implementar uma função derivadas para as caracteristicas
        # --- Entropia espectral -----------------------------------------------
        if (not self.features["spc_entropy"].computed):
            if (self.features["spectogram"].computed):
                data_spectogram = self.features["spectogram"].data
            prob_mtx = np.empty(data_spectogram.shape)
            for idx in range (0,h_FFT):
                X = data_spectogram[idx,:][:, np.newaxis]    
                kde = KernelDensity(kernel='gaussian').fit(X)
                prob_mtx[idx,:] = np.exp(kde.score_samples(X))
                entropy_mtx = np.sum(np.multiply(prob_mtx,np.log(prob_mtx)),axis=0)        
        # ----------------------------------------------------------------------
        # --- CONTINUA MFCC
        if (not self.features["mfcc"].computed):
            d_mfcc_spec = computeD(mfcc_spec)
            dd_mfcc_spec = computeD(d_mfcc_spec)
            mfcc_spec = np.concatenate((mfcc_spec,d_mfcc_spec,dd_mfcc_spec),axis=0)
        # --- CONTINUA PNCC
        if (not self.features["pncc"].computed):
            d_pncc_spec = computeD(pncc_spec)
            dd_pncc_spec = computeD(d_pncc_spec)
            pncc_spec = np.concatenate((pncc_spec,d_pncc_spec,dd_pncc_spec),axis=0)
        
        # --- CONTINUA PLP e RASTA-PLP
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
        
        # ======================================================================
        # TODO implementar uma função de check para as caracteristicas
        self.features["time_domain"] = data_time
        self.features["freq_domain"] = data_freq_domain
        if (not self.features["mfec"].computed):
            self.features["mfec"].data = mfec_spec
            self.features["mfec"].computed = True   
        if (not self.features["teocc"].computed):
            self.features["teocc"].data = teocc_spec
            self.features["teocc"].computed = True   
        if (not self.features["vad_sonh"].computed):
            self.features["vad_sonh"].data = vad_sonh
            self.features["vad_sonh"].computed = True    
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
        if (not self.features["mfcc"].computed):
            self.features["mfcc"].data = mfcc_spec
            self.features["mfcc"].computed = True
        if (not self.features["pncc"].computed):
            self.features["pncc"].data = pncc_spec
            self.features["pncc"].computed = True  
        # ======================================================================    
            
            
            
            
            
