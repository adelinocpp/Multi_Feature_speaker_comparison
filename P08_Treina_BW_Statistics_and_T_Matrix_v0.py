#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:01:58 2022

@author: adelino
"""
import sys
import config as c
import dill
import os
from imports.compute_bw_stats import compute_bw_stats
from imports.train_tv_space import train_tv_space
import numpy as np
from sklearn.preprocessing import StandardScaler
from imports.files_utils import list_contend
# -----------------------------------------------------------------------------        
Compute_BW_stats_T_matrix = True

Compute_ivector = True
# Compute_xvector_TDDNN = True
# Compute_xvector_RESNET = True

# -----------------------------------------------------------------------------
if (not os.path.exists(c.CALIBRATION_DIR)):
    print("Diretório de características de calibração não existe. Executar as rotinas P01 e P02.")
    Compute_BW_stats_T_matrix = False        

if (not os.path.exists(c.VALIDATION_DIR)):
    print("Diretório de características de validação não existe. Executar as rotina P01 e  P02.")
    Compute_BW_stats_T_matrix = False      
    
if (not os.path.exists(c.GMM_UBM_FILE_NAME)):
    print("Arquivo UBM não existe. Executar a rotina P03.")
    Compute_BW_stats_T_matrix = False      
    
# -----------------------------------------------------------------------------
if (Compute_BW_stats_T_matrix and Compute_ivector):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
        
    ComputeStatsTrain = not os.path.exists(c.T_MATRIX_STATS_TRAIN_FILE_IVECTOR);
    
    if (ComputeStatsTrain):
        pattern = ('.p',)
        file_list = list_contend(c.CALIBRATION_DIR,pattern)
        file_list.sort()
        statsTrain = np.zeros((len(file_list),UBM.n_components*(UBM.n_features_in_ + 1) ) )
        for idx, filename in enumerate(file_list):
            ofile = open(filename, "rb")
            featureObj = dill.load(ofile)
            ofile.close()
            # Features aready filtered: set -> filter_vad=True
            features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
            
            N, F = compute_bw_stats(features.T, UBM);
            statsTrain[idx,:] = np.append(N,F)
            
            print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
            
        ofile = open(c.T_MATRIX_STATS_TRAIN_FILE_IVECTOR, "wb")
        dill.dump(statsTrain, ofile)
        ofile.close()
    else:
        ofile = open(c.T_MATRIX_STATS_TRAIN_FILE_IVECTOR, "rb")
        statsTrain = dill.load(ofile)
        ofile.close()
    
    print('Iniciando calculo da matrix T... ')
    T_matrix = train_tv_space(statsTrain, UBM, c.T_MATRIX_DIM, c.T_MATRIX_MAX_ITER, c.T_MATRIX_FILE_IVECTOR)
    print('Calculo da matrix T finalizado.')
# -----------------------------------------------------------------------------
