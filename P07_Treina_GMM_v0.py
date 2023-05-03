#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:58:00 2021

@author: adelino
"""
import sys
import config as c
import dill
import os
from sklearn.mixture import GaussianMixture
import numpy as np
from imports.files_utils import list_contend, build_folders_to_save
# -----------------------------------------------------------------------------
def mapAdapt(vX, UBM):
    posterioriX = UBM.predict_proba(vX)    
    mapAdaptRelevance = 19
    vec_sumPosP = posterioriX.sum(0)
    vec_sumPosMu = np.matmul(posterioriX.T,vX)
    vec_sumPosDSig = np.matmul(posterioriX.T,np.power(vX,2))
    sum_N = vec_sumPosP.sum(0)
    sum_W = 0
    m_Rho = np.zeros(UBM.n_components)
    m_Mu = np.zeros([UBM.n_components, vX.shape[1]])
    m_Sig = np.zeros([UBM.n_components, vX.shape[1]])
    for k in range(0,UBM.n_components):
        dbl_alpha = vec_sumPosP[k]/(vec_sumPosP[k] + mapAdaptRelevance)
        dbl_Wtemp = UBM.weights_[k]*(1-dbl_alpha)  + dbl_alpha*vec_sumPosP[k]/sum_N
        sum_W += dbl_Wtemp
        m_Rho[k] = dbl_Wtemp
        m_Mu[k,:] = UBM.means_[k,:]*(1-dbl_alpha) + dbl_alpha*vec_sumPosMu[k,:]/(vec_sumPosP[k] + 1e-12)
        m_Sig[k,:] = (UBM.covariances_[k,:] + np.power(UBM.means_[k,:],2))*(1-dbl_alpha) + \
                        dbl_alpha*vec_sumPosDSig[k,:]/(vec_sumPosP[k] + 1e-12) - np.power(m_Mu[k,:],2)
    
    m_Rho = m_Rho/(m_Rho.sum(0))
    GMM = GaussianMixture(n_components = UBM.n_components, covariance_type=UBM.covariance_type)
    GMM.weights_ = m_Rho
    GMM.means_ = m_Mu
    GMM.covariances_ = m_Sig
    GMM.precisions_ = 1/m_Sig
    GMM.precisions_cholesky_ = 1/m_Sig
    return GMM
# -----------------------------------------------------------------------------        
Compute_GMM_Train = False
Compute_GMM_Test = False
Compute_GMM_UBM = True

if (not os.path.exists(c.CALIBRATION_DIR)):
    print("Diretório de características de calibração não existe. Executar as rotinas P01 e P02.")
    Compute_GMM_Train = False

if (not os.path.exists(c.VALIDATION_DIR)):
    print("Diretório de características de validação não existe. Executar as rotina P01 e  P02.")
    Compute_GMM_Train = False
    
if (not os.path.exists(c.GMM_UBM_FILE_NAME)):
    print("Arquivo UBM não existe. Executar a rotina P03.")
    Compute_GMM_Train = False
    Compute_GMM_Test = True
    
if (Compute_GMM_Train):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
    pattern = ('.p',)
    file_list = list_contend(c.CALIBRATION_DIR,pattern)
    file_list.sort()
    
    print('Inicio do calculo dos GMM:')
    print('Arquivos de calibração: {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
                    
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.GMM_MODEL_CALIBRATION_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        
        gmm = {}
        gmm['model'] = mapAdapt(features.T,UBM)
        
        build_folders_to_save(filenameSave)
        
        ofile = open(filenameSave, "wb")
        dill.dump(gmm, ofile)
        ofile.close()
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
    
if (Compute_GMM_Test):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
    
    pattern = ('.p',)
    file_list = list_contend(c.VALIDATION_DIR,pattern)
    file_list.sort()
            
    print('Inicio do calculo dos GMM:')
    print('Arquivos de teste: {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
            
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.GMM_MODEL_VALIDATION_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        
        gmm = {}
        gmm['model'] = mapAdapt(features.T,UBM)
        
        build_folders_to_save(filenameSave)
            
        ofile = open(filenameSave, "wb")
        dill.dump(gmm, ofile)
        ofile.close()
            
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        # if (DEBUG_MODE):
        #     sys.exit("MODO DEPURACAO: Fim do script")
    
if (Compute_GMM_UBM):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
    
    pattern = ('.p',)
    file_list = list_contend(c.UBM_DIR,pattern)
    file_list.sort()
            
    print('Inicio do calculo dos GMM:')
    print('Arquivos de UBM: {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        ofile = open(filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False).astype(np.float64)
            
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.GMM_MODEL_UBM_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        
        gmm = {}
        gmm['model'] = mapAdapt(features.T,UBM)
        
        build_folders_to_save(filenameSave)
                    
        ofile = open(filenameSave, "wb")
        dill.dump(gmm, ofile)
        ofile.close()
            
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        # if (DEBUG_MODE):
        #     sys.exit("MODO DEPURACAO: Fim do script")