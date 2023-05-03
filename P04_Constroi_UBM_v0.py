#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:35:48 2022

@author: adelino
"""
import config as c
import os
import dill
import sys
import gc
from imports import welford
from imports.files_utils import list_contend
import numpy as np
from imports.acoustic_features import Feature
from imports.files_utils import build_folders_to_save
from imports.acoustic_features import AcousticsFeatures
import math
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------
def normalize_frames(m,ubm_mean,ubm_std):
    return (m - ubm_mean) / (ubm_std + 2e-12)
# ----------------------------------------------------------------------------
gc.enable()

Compute_Train_UBM_Mean_Std = True
Only_Compute_GMM_UBM_Train = False
Normalise_Other_Pools = True
if (not os.path.exists(c.DB_TIME_SEPR_FILE)):
    print("Arquivo {:} não existe. Executar a rotina P02.".format(c.DB_TIME_SEPR_FILE))
    Compute_GMM_UBM_Train = False
    Compute_Train_UBM_Mean_Std = False
    sys.exit("Erro ao carregar arquivo.")
else:
    ofile = open(c.DB_TIME_SEPR_FILE, "rb")
    listPool = dill.load(ofile)
    ofile.close()
    print("Arquivo de tempo \"{:}\" carregado.".format(c.DB_TIME_SEPR_FILE))    

if (not os.path.exists(c.FEATURE_FILE_LIST)):
    pattern = (c.RAW_FEAT_EXT,)
    file_list = list_contend(c.FEATURE_FILE_OUTPUT,pattern)
    ofile = open(c.FEATURE_FILE_LIST, "wb")
    dill.dump(file_list, ofile)
    ofile.close()
else:
    ofile = open(c.FEATURE_FILE_LIST, "rb")
    file_list = dill.load(ofile)
    ofile.close()
    print("Arquivo de lista de caractersiticas \"{:}\" carregado.".format(c.FEATURE_FILE_LIST)) 

if (listPool[1].pool_name == 'UBM'):
    indexList = np.sort(listPool[1].pool_speakers)

umbIndex = np.array([filename.split("/")[3] in indexList for filename in file_list]).nonzero()[0]
selFileList = np.array(file_list)[umbIndex]


if (not os.path.exists(c.UBM_FILE_NAME)) and (Compute_Train_UBM_Mean_Std):
    w_ubm = welford.Welford()
    print('Inicio do calculo do UBM:')
    print('Arquivos de treinamento: {}'.format(len(selFileList)))
    for idx, filename in enumerate(selFileList):
        ofile = open(filename, "rb")
        speakerObj = dill.load(ofile)
        ofile.close()
        # ---------------------------------------------------------------------
        featureObj = speakerObj.features["mfcc"]
        # --- Esqueci de calcular na rotina P01, melhor que economiza espaço
        if ((not featureObj.has_delta) and (not featureObj.has_deltadelta)):
            speakerObj.apply_delta("mfcc")
        features = speakerObj.get_feature_by_name("mfcc").astype(np.float64)
        '''
        # --- Esqueci de calcular na rotina P01, melhor que economiza espaço --
        speakerObj.apply_delta("mfcc")
        # ---------------------------------------------------------------------
        features = speakerObj.get_feature_by_name("mfcc")
        '''
        w_ubm.update(features.T)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(selFileList)-1));
    ubm_data = {}
    ubm_data["mean"]    = w_ubm.mean
    ubm_data["std"]     = w_ubm.std
    ofile = open(c.UBM_FILE_NAME, "wb")
    dill.dump(ubm_data, ofile)
    ofile.close()

elif(os.path.exists(c.UBM_FILE_NAME)):
    ofile = open(c.UBM_FILE_NAME, "rb")
    ubm_data = dill.load(ofile)
    ofile.close()
    print("Arquivo \"{:}\" carregado.".format(c.UBM_FILE_NAME))     

if (Normalise_Other_Pools):
    print('Normalização das demais características de treinamento...')
    for pool in listPool:
        indexList = np.sort(pool.pool_speakers)
        poolIndex = np.array([filename.split("/")[3] in indexList for filename in file_list]).nonzero()[0]
        selFileList = np.array(file_list)[poolIndex]
        if (pool.pool_name == 'UBM'):
            if not (os.path.exists(c.GMM_UBM_DATA_FILE_NAME)):
                ubmData = np.empty(shape=[0, len(ubm_data["mean"])]).astype(np.float64)
                print("--- POOL: {:} ---------------------".format(pool.pool_name))
                for idx, filename in enumerate(selFileList):
                    ofile = open(filename, "rb")
                    speakerObj = dill.load(ofile)
                    ofile.close()
                    # ---------------------------------------------------------------------
                    featureObj = speakerObj.features["mfcc"]
                    # --- Esqueci de calcular na rotina P01, melhor que economiza espaço
                    if ((not featureObj.has_delta) and (not featureObj.has_deltadelta)):
                        speakerObj.apply_delta("mfcc")
                    features = speakerObj.get_feature_by_name("mfcc").astype(np.float64)
                    if (not featureObj.normalized):
                        features = normalize_frames(features.T,\
                               ubm_data["mean"].astype(np.float64),ubm_data["std"].astype(np.float64))
                    
                    features = features.astype(np.float64)
                    ubmData = np.append(ubmData,  features, axis=0)
                    print('Carregado arquivo {:4} de {:4}'.format(idx, len(selFileList)-1));

                ofile = open(c.GMM_UBM_DATA_FILE_NAME, "wb")
                dill.dump(ubmData, ofile)
                ofile.close()
            else:
                ofile = open(c.GMM_UBM_DATA_FILE_NAME, "rb")
                ubmData = dill.load(ofile)
                ofile.close()
                ubmData = ubmData.astype(np.float64)
                print('Carregado arquivo {:4}.'.format(c.GMM_UBM_DATA_FILE_NAME))

            print("Treinando UBM! ubmData {:} x {:}".format(ubmData.shape[0], ubmData.shape[1]))
            infiniteMtx = (np.isfinite(ubmData)==False).astype(np.int32)
            n_NAN = np.sum(infiniteMtx)        
            if (n_NAN > 0):
                idxFinite = (np.sum(infiniteMtx,axis=1) == 0).nonzero()[0]
                print("idx finite {:}".format(idxFinite.shape))
                print("Remove UBM not finite {:} length {:}".format(n_NAN,len(idxFinite)))
                ubmData = ubmData[:,idxFinite].astype(np.float64)
                ofile = open(c.GMM_UBM_DATA_FILE_NAME, "wb")
                dill.dump(ubmData, ofile)
                ofile.close()

            nFrames = ubmData.shape[0]
            ds_factor = 8
            ubmIniPow = 0
            if (os.path.exists(c.GMM_UBM_FILE_NAME)):
                print("Modelo UBM em desenvolvimento, carregando UBM...")
                ofile = open(c.GMM_UBM_FILE_NAME, "rb")
                UBM = dill.load(ofile)
                ofile.close()
                print("Carregado UBM com {:} componentes...".format(UBM.n_components))
                ubmIniPow = int(np.log2(UBM.n_components)) + 1;
            else:
                print("Iniciando outro modelo UBM...")
            k_ds = 2
            for nPow in range(ubmIniPow,int(1+math.log2(c.UBM_nComponents))):
                nComp = 2 ** nPow
                if (nComp > 64):
                    ds_factor = ds_factor*k_ds
                    #k_ds =  k_ds + 1
                print('Iniciando com {} componentes. Dow sample k {:}'.format(nComp,ds_factor))
                if not ("ubmData" in locals()):
                    ofile = open(c.GMM_UBM_DATA_FILE_NAME, "rb")
                    ubmData = dill.load(ofile)
                    ofile.close()
                if (nComp > c.UBM_nComponents/2.1 ):
                    idxSel = np.random.permutation(nFrames)[0:int(nFrames/ds_factor)]
                    idxSel.sort()
                    regData = ubmData[idxSel,:].astype(np.float32) # era float64
                else:
                    regData = ubmData.astype(np.float32)

                del ubmData
                print("Release memory: {:}".format(gc.collect()))
                
                if (nComp == 1):
                    UBM = GaussianMixture(n_components = nComp, covariance_type=c.UBM_covType, 
                                          reg_covar=1e-6,init_params='kmeans++', n_init=10, 
                                          tol=1e-3, verbose = 2, max_iter=200).fit(regData)
                else:
                    epsM = np.zeros(UBM.means_.shape).astype(np.float32)
                    idxMaxPrec = np.argmax(UBM.covariances_.max(0))
                    epsM[:,idxMaxPrec] = np.sqrt(np.max(UBM.covariances_.max(0)))*np.ones([1,UBM.precisions_.shape[0]]).astype(np.float32)
                    wIni = 0.5*np.append(UBM.weights_,UBM.weights_,axis=0).astype(np.float32)
                    wIni = wIni/np.sum(wIni)
                    mIni = np.append(UBM.means_.astype(np.float32) - epsM,UBM.means_.astype(np.float32) + epsM,axis=0)
                    pIni = np.append(UBM.precisions_.astype(np.float64),UBM.precisions_.astype(np.float64),axis=0)
                    UBM = GaussianMixture(n_components = nComp, covariance_type=c.UBM_covType,
                                          weights_init= wIni,
                                          means_init=mIni,
                                          precisions_init=pIni,
                                          reg_covar=1e-6, max_iter=200, n_init=1,
                                          tol=1e-4, verbose = 2).fit(regData)
                    
                ofile = open(c.GMM_UBM_FILE_NAME, "wb")
                dill.dump(UBM, ofile)
                ofile.close()
    
        if (pool.pool_name == 'LDA') and (not Only_Compute_GMM_UBM_Train):
            print("LDA")
            lastSpeakerID = '0'
            nFiles = 0;
            for idx, filename in enumerate(selFileList):
                if (idx < pool.pool_prop[0]):
                    nDiv = 3
                elif(idx >= pool.pool_prop[0]) and (idx < (pool.pool_prop[0] + pool.pool_prop[1])):
                    nDiv = 5
                else:
                    nDiv = 7
                currSpeakerID = filename.split("/")[-2]
                ofile = open(filename, "rb")
                speakerObj = dill.load(ofile)
                ofile.close()
                # ---------------------------------------------------------------------
                featureObj = speakerObj.features["mfcc"]
                # --- Esqueci de calcular na rotina P01, melhor que economiza espaço
                if ((not featureObj.has_delta) and (not featureObj.has_deltadelta)):
                    speakerObj.apply_delta("mfcc")
                features = speakerObj.get_feature_by_name("mfcc").astype(np.float64)
                if (not featureObj.normalized):
                    features = normalize_frames(features.T,\
                           ubm_data["mean"].astype(np.float64),ubm_data["std"].astype(np.float64))
                '''
                # --- Esqueci de calcular na rotina P01, melhor que economiza espaço
                speakerObj.apply_delta("mfcc",Force =True)
                # ---------------------------------------------------------------------
                features = speakerObj.get_feature_by_name("mfcc").astype(np.float64)
                featureObj = speakerObj.features["mfcc"]
                features = normalize_frames(features.T,\
                    np.array(ubm_data["mean"],np.float64),np.array(ubm_data["std"],np.float64))
                '''    
                if not (currSpeakerID == lastSpeakerID):
                    nJoinFeatures = features
                    nFiles = 1
                    lastSpeakerID = currSpeakerID
                else:
                    nJoinFeatures = np.append(nJoinFeatures,features, axis=0)
                    nFiles += 1
                    
                if (nFiles == 2):
                    nFrames = nJoinFeatures.shape[0]
                    for iDiv in range(0, nDiv):
                        idxIni = int(np.ceil(iDiv/nDiv*nFrames))
                        idxFim = np.min([idxIni + int(np.ceil(nFrames/nDiv)), nFrames])
                        norm_feature_obj = Feature(data=nJoinFeatures[idxIni:idxFim,:].T,
                                                   computed=True,name="mfcc",normalized=True,
                                                   has_delta=True,has_deltadelta=True)
                        # norm_feature_obj.get_delta_status(featureObj)
                        speakerObj.features["mfcc"] = norm_feature_obj
                        speakerObj.feature_file = "{:}{:}/{:}/LDA_{:02}.p".format(c.DB_POOL_EXPERIMENT,pool.pool_name,filename.split("/")[3],iDiv)
                        build_folders_to_save(speakerObj.feature_file )
                        ofile = open(speakerObj.get_feature_file(), "wb")
                        dill.dump(speakerObj, ofile)
                        ofile.close()
                    nFiles = 0
                print('Finalizado arquivo {:4} de {:4}'.format(idx, len(selFileList)-1));
                # if (idx > 3):
                #     sys.exit("MODO DEPURACAO: Fim do script")
            
            #     print('Finalizado arquivo {:4} de {:4}'.format(idx, len(selFileList)-1));
        elif (not Only_Compute_GMM_UBM_Train):
            print("--- POOL: {:} ---------------------".format(pool.pool_name))
            lastSpeakerID = '0'
            nFiles = 0;
            for idx, filename in enumerate(selFileList):
                nDiv = 2
                currSpeakerID = filename.split("/")[-2]
                ofile = open(filename, "rb")
                speakerObj = dill.load(ofile)
                ofile.close()
                # ---------------------------------------------------------------------
                featureObj = speakerObj.features["mfcc"]
                # --- Esqueci de calcular na rotina P01, melhor que economiza espaço
                if ((not featureObj.has_delta) and (not featureObj.has_deltadelta)):
                    speakerObj.apply_delta("mfcc")
                features = speakerObj.get_feature_by_name("mfcc").astype(np.float64)
                if (not featureObj.normalized):
                    features = normalize_frames(features.T,\
                           ubm_data["mean"].astype(np.float64),ubm_data["std"].astype(np.float64))
                '''
                # --- Esqueci de calcular na rotina P01, melhor que economiza espaço
                speakerObj.apply_delta("mfcc",Force =True)
                # ---------------------------------------------------------------------
                features = speakerObj.get_feature_by_name("mfcc").astype(np.float64)
                featureObj = speakerObj.features["mfcc"]
                features = normalize_frames(features.T,\
                    np.array(ubm_data["mean"],np.float64),np.array(ubm_data["std"],np.float64))
                '''    
                if not (currSpeakerID == lastSpeakerID):
                    nJoinFeatures = features
                    nFiles = 1
                    lastSpeakerID = currSpeakerID
                else:
                    nJoinFeatures = np.append(nJoinFeatures,features, axis=0)
                    nFiles += 1
                
                if (nFiles == 2):
                    nFrames = nJoinFeatures.shape[0]
                    for iDiv in range(0, nDiv):
                        idxIni = int(np.ceil(iDiv/nDiv*nFrames))
                        idxFim = np.min([idxIni + int(np.ceil(nFrames/nDiv)), nFrames])
                        norm_feature_obj = Feature(data=nJoinFeatures[idxIni:idxFim,:].T,
                                                   computed=True,name="mfcc",normalized=True,
                                                   has_delta=True,has_deltadelta=True)
                        # norm_feature_obj.get_delta_status(featureObj)
                        speakerObj.features["mfcc"] = norm_feature_obj
                        speakerObj.feature_file = "{:}{:}/{:}/{:}_{:02}.p".format(c.DB_POOL_EXPERIMENT,\
                                                pool.pool_name,filename.split("/")[3],pool.pool_name,iDiv)
                        build_folders_to_save(speakerObj.feature_file )
                        ofile = open(speakerObj.get_feature_file(), "wb")
                        dill.dump(speakerObj, ofile)
                        ofile.close()
                    nFiles = 0
                    
                # norm_feature_obj = Feature(data=features.T,computed=True,name="mfcc")

                # speakerObj.features["mfcc"] = norm_feature_obj
                
                # speakerObj.feature_file = "{:}{:}/{:}".format(c.DB_POOL_EXPERIMENT,pool.pool_name,"/".join(filename.split("/")[3:]))
                # build_folders_to_save(speakerObj.feature_file )
                # ofile = open(speakerObj.get_feature_file(), "wb")
                # dill.dump(speakerObj, ofile)
                # ofile.close()
                
                print('Finalizado arquivo {:4} de {:4}'.format(idx, len(selFileList)-1));
            #     sys.exit("MODO DEPURACAO: Fim do script")
            
