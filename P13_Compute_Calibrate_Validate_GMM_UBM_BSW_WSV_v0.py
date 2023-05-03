# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:08:15 2022

@author: adelino
"""
import config as c
import sys
import dill
import numpy as np
import os
from imports.files_utils import list_contend
# -----------------------------------------------------------------------------        
Compute_GMM_compute_data = True
Compute_GMM_Validate_Data = True


if (Compute_GMM_compute_data and (not os.path.exists(c.GMM_CALIBRATE_FILEDATA))):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
    pattern = ('.p',)
    feature_file_list = list_contend(c.CALIBRATION_DIR,pattern)
    feature_file_list.sort()    
    gmm_ubm_files_list = list_contend(c.GMM_MODEL_UBM_DIR,pattern)
    gmm_ubm_files_list.sort()
    gmm_calibrate_files_list = list_contend(c.GMM_MODEL_CALIBRATION_DIR,pattern)
    gmm_calibrate_files_list.sort()
    
    gmm_compute_data = {}
    print('Etapa de calibracao')
    print('Confronto das carcterísticas de cada arquivo com modelos UBM:')
    print('Arquivos de caracteristicas: {}'.format(len(feature_file_list)))
    print('Arquivos de GMMs WSV: {}'.format(len(gmm_ubm_files_list)))
    print('Arquivos de GMMs BSV: {}'.format(len(gmm_calibrate_files_list)))
    gmm_calibrate_list = []
    for idxF, feature_filename in enumerate(feature_file_list):
        gmm_compute_data = {}
        file_order = int(feature_filename.split('/')[-1].split('.')[0].split('_')[1])
        if (file_order == 1):
            continue
        feature_id = feature_filename.split('/')[-2]
        ofile = open(feature_filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False)
        predUBM = UBM.score_samples(features.T)
        listScoreWSV = []
        listWSVFileComp = []
        for idxG, gmm_filename in enumerate(gmm_ubm_files_list):
            file_order = int(gmm_filename.split('/')[-1].split('.')[0].split('_')[1])
            if (file_order == 1):
                continue
            gmm_id = gmm_filename.split('/')[-2]
            ofile = open(gmm_filename, "rb")
            gmm = dill.load(ofile)
            ofile.close()
            GMM = gmm['model']
            predGMM = GMM.score_samples(features.T)
            mtxScore = np.mean(predGMM - predUBM)
            listScoreWSV.append(mtxScore)
            listWSVFileComp.append(gmm_id)
            if ((idxG % 100) == 0):
                print("GMM UBM n {:} de {:}".format(idxG,len(gmm_ubm_files_list)-1))
        
        listScoreBSV = []
        listBSVFileComp = []
        matchSpeaker = []
        for idxC, calibrate_filename in enumerate(gmm_calibrate_files_list):
            file_order = int(calibrate_filename.split('/')[-1].split('.')[0].split('_')[1])
            if (file_order == 0):
                continue
            
            calibrate_id = calibrate_filename.split('/')[-2]
            ofile = open(calibrate_filename, "rb")
            gmm = dill.load(ofile)
            ofile.close()
            GMM = gmm['model']
            predGMM = GMM.score_samples(features.T)
            if (calibrate_id == feature_id):
                speaker_comp_score = predGMM
            mtxScore = np.mean(predGMM - predUBM)
            listScoreBSV.append(mtxScore)
            listBSVFileComp.append(calibrate_id)
            matchSpeaker.append(int(calibrate_id == feature_id))
            if ((idxC % 100) == 1):
                print("GMM CALIBRATE n {:} de {:}".format(idxC,len(gmm_calibrate_files_list)-1))
        print("Fim do arquivo caracterisisticas {:} de {:}".format(idxF,len(feature_file_list)-1))
        
        gmm_compute_data["Speaker_id"] = feature_id
        gmm_compute_data["speaker_comp_score"] = speaker_comp_score
        gmm_compute_data["speaker_ubm_score"] = predUBM
        gmm_compute_data["speaker_BSV"] = listScoreBSV
        gmm_compute_data["speaker_WSV"] = listScoreWSV
        gmm_compute_data["listBSVFileComp"] = listBSVFileComp
        gmm_compute_data["listWSVFileComp"] = listWSVFileComp
        gmm_compute_data["matchSpeaker"] = matchSpeaker
        gmm_calibrate_list.append(gmm_compute_data)
    ofile = open(c.GMM_CALIBRATE_FILEDATA, "wb")
    dill.dump(gmm_calibrate_list, ofile)
    ofile.close()

# =============================================================================
if (Compute_GMM_Validate_Data and (not os.path.exists(c.GMM_VALIDATE_FILEDATA))):
    ofile = open(c.GMM_UBM_FILE_NAME, "rb")
    UBM = dill.load(ofile)
    ofile.close()
    pattern = ('.p',)
    feature_file_list = list_contend(c.VALIDATION_DIR,pattern)
    feature_file_list.sort()    
    gmm_ubm_files_list = list_contend(c.GMM_MODEL_UBM_DIR,pattern)
    gmm_ubm_files_list.sort()
    gmm_calibrate_files_list = list_contend(c.GMM_MODEL_VALIDATION_DIR,pattern)
    gmm_calibrate_files_list.sort()
    
    print('Etapa de validacao')
    print('Confronto das carcterísticas de cada arquivo com modelos UBM:')
    print('Arquivos de caracteristicas: {}'.format(len(feature_file_list)))
    print('Arquivos de GMMs WSV: {}'.format(len(gmm_ubm_files_list)))
    print('Arquivos de GMMs BSV: {}'.format(len(gmm_calibrate_files_list)))
    gmm_validate_list = []
    for idxF, feature_filename in enumerate(feature_file_list):
        gmm_compute_data = {}
        file_order = int(feature_filename.split('/')[-1].split('.')[0].split('_')[1])
        if (file_order == 1):
            continue
        feature_id = feature_filename.split('/')[-2]
        ofile = open(feature_filename, "rb")
        featureObj = dill.load(ofile)
        ofile.close()
        # Features aready filtered: set -> filter_vad=True
        features = featureObj.get_feature_by_name("mfcc",filter_vad=False)
        predUBM = UBM.score_samples(features.T)
        listScoreWSV = []
        listWSVFileComp = []
        for idxG, gmm_filename in enumerate(gmm_ubm_files_list):
            file_order = int(gmm_filename.split('/')[-1].split('.')[0].split('_')[1])
            if (file_order == 1):
                continue
            gmm_id = gmm_filename.split('/')[-2]
            ofile = open(gmm_filename, "rb")
            gmm = dill.load(ofile)
            ofile.close()
            GMM = gmm['model']
            predGMM = GMM.score_samples(features.T)
            mtxScore = np.mean(predGMM - predUBM)
            listScoreWSV.append(mtxScore)
            listWSVFileComp.append(gmm_id)
            if ((idxG % 100) == 0):
                print("GMM UBM n {:} de {:}".format(idxG,len(gmm_ubm_files_list)-1))
            
        listScoreBSV = []
        listBSVFileComp = []
        matchSpeaker = []
        for idxC, calibrate_filename in enumerate(gmm_calibrate_files_list):
            file_order = int(calibrate_filename.split('/')[-1].split('.')[0].split('_')[1])
            if (file_order == 0):
                continue
            
            calibrate_id = calibrate_filename.split('/')[-2]
            ofile = open(calibrate_filename, "rb")
            gmm = dill.load(ofile)
            ofile.close()
            GMM = gmm['model']
            predGMM = GMM.score_samples(features.T)
            if (calibrate_id == feature_id):
                speaker_comp_score = predGMM
            mtxScore = np.mean(predGMM - predUBM)
            listScoreBSV.append(mtxScore)
            listBSVFileComp.append(calibrate_id)
            matchSpeaker.append(int(calibrate_id == feature_id))
            if ((idxC % 100) == 1):
                print("GMM CALIBRATE n {:} de {:}".format(idxC,len(gmm_calibrate_files_list)-1))
        print("Fim do arquivo caracterisisticas {:} de {:}".format(idxF,len(feature_file_list)-1))
        
        gmm_compute_data["Speaker_id"] = feature_id
        gmm_compute_data["speaker_comp_score"] = speaker_comp_score
        gmm_compute_data["speaker_ubm_score"] = predUBM
        gmm_compute_data["speaker_BSV"] = listScoreBSV
        gmm_compute_data["speaker_WSV"] = listScoreWSV
        gmm_compute_data["listBSVFileComp"] = listBSVFileComp
        gmm_compute_data["listWSVFileComp"] = listWSVFileComp
        gmm_compute_data["matchSpeaker"] = matchSpeaker
        gmm_validate_list.append(gmm_compute_data)
    ofile = open(c.GMM_VALIDATE_FILEDATA, "wb")
    dill.dump(gmm_validate_list, ofile)
    ofile.close()
