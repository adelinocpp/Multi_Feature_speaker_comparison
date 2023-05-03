#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:09:42 2022

@author: adelino
"""
# -----------------------------------------------------------------------------
import sys
import config as c
import pickle
import dill
import os
from imports.files_utils import list_contend, build_folders_to_save
import numpy as np
from imports.compute_bw_stats import compute_bw_stats
from imports.extract_ivector import extract_ivector
# -----------------------------------------------------------------------------
Compute_i_vectors = True

if (not os.path.exists(c.CALIBRATION_DIR)):
    print("Diretório de características de calibração não existe. Executar as rotinas P01 e P02.")
    Compute_i_vectors = False          

if (not os.path.exists(c.VALIDATION_DIR)):
    print("Diretório de características de validação não existe. Executar as rotina P01 e  P02.")
    Compute_i_vectors = False        
    
if (not os.path.exists(c.GMM_UBM_FILE_NAME)):
    print("Arquivo UBM não existe. Executar a rotina P03.")
    Compute_i_vectors = False           
    
if (not os.path.exists(c.T_MATRIX_FILE_IVECTOR)):
    print("Arquivo da matriz de variabildiade total (matrix T) não existe. Executar a rotina P05.")
    Compute_i_vectors = False
   
if (Compute_i_vectors):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
    
    with open(c.T_MATRIX_FILE_IVECTOR, 'rb') as f:
        T = pickle.load(f)
    
    pattern = ('.p',)
    
    # -------------------------------------------------------------------------
    # First - Gera os i-vectors de calibracao
    # -------------------------------------------------------------------------
    file_list = list_contend(c.CALIBRATION_DIR, pattern = ('.p',))
    file_list.sort()
    print("Iniciado i-vector calibracao...")
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
        N, F = compute_bw_stats(features.T, UBM);
        x = extract_ivector(np.append(N,F),UBM,T)
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        out_file_name = filenameParts.split('/')[-1].split('.')[0]
        
        filenameSave = c.IVECTOR_MODEL_CALIBRATION_DIR + '{:}/{:}'.format(filenameFolder,out_file_name)+'.pth'
        build_folders_to_save(filenameSave)
        
        ofile = open(filenameSave, "wb")
        dill.dump(x, ofile)
        ofile.close()
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Second - Gera os i-vectors de validacao
    # -------------------------------------------------------------------------
    file_list = list_contend(c.VALIDATION_DIR, pattern = ('.p',))
    file_list.sort()
    print("Iniciado i-vector validacao...")
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
        N, F = compute_bw_stats(features.T, UBM);
        x = extract_ivector(np.append(N,F),UBM,T)
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        out_file_name = filenameParts.split('/')[-1].split('.')[0]
        
        filenameSave = c.IVECTOR_MODEL_VALIDATION_DIR + '{:}/{:}'.format(filenameFolder,out_file_name)+'.pth'
        build_folders_to_save(filenameSave)
        
        ofile = open(filenameSave, "wb")
        dill.dump(x, ofile)
        ofile.close()
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Last - Gera os i-vectors para UBM
    # -------------------------------------------------------------------------
    file_list = list_contend(c.UBM_DIR, pattern = ('.p',))
    file_list.sort()
    print("Iniciado i-vector ubm...")
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
        N, F = compute_bw_stats(features.T, UBM);
        x = extract_ivector(np.append(N,F),UBM,T)
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        out_file_name = filenameParts.split('/')[-1].split('.')[0]
        
        filenameSave = c.IVECTOR_MODEL_UBM_DIR + '{:}/{:}'.format(filenameFolder,out_file_name)+'.pth'
        build_folders_to_save(filenameSave)
        
        ofile = open(filenameSave, "wb")
        dill.dump(x, ofile)
        ofile.close()
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Last - Gera os i-vectors para LDA
    # -------------------------------------------------------------------------
    file_list = list_contend(c.LDA_DIR, pattern = ('.p',))
    file_list.sort()
    print("Iniciado i-vector lda...")
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
        N, F = compute_bw_stats(features.T, UBM);
        x = extract_ivector(np.append(N,F),UBM,T)
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        out_file_name = filenameParts.split('/')[-1].split('.')[0]
        
        filenameSave = c.IVECTOR_MODEL_LDA_DIR + '{:}/{:}'.format(filenameFolder,out_file_name)+'.pth'
        build_folders_to_save(filenameSave)
        
        ofile = open(filenameSave, "wb")
        dill.dump(x, ofile)
        ofile.close()
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
    # -------------------------------------------------------------------------
