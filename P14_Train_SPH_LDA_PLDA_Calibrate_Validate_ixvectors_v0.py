#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:56:06 2022

@author: adelino
"""
import config as c
from imports.files_utils import list_contend
import dill
import numpy as np
from imports.SpheringSVD import SpheringSVD as Sphering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import imports.plda as plda
import os

LDA_IVECTOR_RUN = True
LDA_IVECTOR_CALIBRATE_RUN = True
LDA_IVECTOR_VALIDATE_RUN = True

LDA_XVECTOR_TDDNN_RUN = True
LDA_XVECTOR_TDDNN_CALIBRATE_RUN = True
LDA_XVECTOR_TDDNN_VALIDATE_RUN = True

LDA_XVECTOR_RESNET_RUN = True
LDA_XVECTOR_RESNET_CALIBRATE_RUN = True
LDA_XVECTOR_RESNET_VALIDATE_RUN = True

pattern = (".pth",)
# =============================================================================
if (LDA_IVECTOR_RUN):
    if (not os.path.exists(c.IVECTOR_SPH_LDA_PLDA_MODELS)):
        model = {}
        # --- usa base LDA ----------------------------------------------------
        lda_ivectos_file_list = list_contend(c.IVECTOR_MODEL_LDA_DIR, pattern)
        nFiles = len(lda_ivectos_file_list)
        y = np.zeros((nFiles,))    
        for idx, file_name in enumerate(lda_ivectos_file_list):
            speaker_id = file_name.split('/')[-2]
            ofile = open(file_name, "rb")
            spk_vector = dill.load(ofile)
            ofile.close()
            y[idx] = int(speaker_id);
            if (idx == 0):
                embedding_size = spk_vector.shape[0]
                X = np.zeros((nFiles,embedding_size))
            X[idx,:] = np.array(spk_vector)
        # --- processo de Sphearing -------------------------------------------
        SpheModel = Sphering.SpheringSVD()
        SpheModel.fit(X)
        Xsph = SpheModel.transform(X)
        LDAModel = LinearDiscriminantAnalysis()
        LDAModel.fit(Xsph, y)
        PLDAModel = plda.Classifier()
        Xlda = LDAModel.transform(Xsph)
        PLDAModel.fit_model(Xlda, y)
        model['sph'] = SpheModel
        model['lda'] = LDAModel
        model['plda'] = PLDAModel
        ofile = open(c.IVECTOR_SPH_LDA_PLDA_MODELS, "wb")
        dill.dump(model, ofile)
        ofile.close()
    else:
        ofile = open(c.IVECTOR_SPH_LDA_PLDA_MODELS, "rb")
        model = dill.load(ofile)
        ofile.close()
        SpheModel = model['sph']
        LDAModel = model['lda']
        PLDAModel = model['plda']
    # =========================================================================
    if (LDA_IVECTOR_CALIBRATE_RUN):
        # --- usa base de calibracao
        ivector_calibration_file_list = list_contend(c.IVECTOR_MODEL_CALIBRATION_DIR,pattern)
        ivector_calibration_file_list.sort()    
        ivector_ubm_file_list = list_contend(c.IVECTOR_MODEL_UBM_DIR,pattern)
        ivector_ubm_file_list.sort()
        print('Etapa de calibracao - ivector')
        print('Arquivos de ivector BSV: {:}'.format(len(ivector_calibration_file_list)))
        print('Arquivos de ivector WSV: {:}'.format(len(ivector_ubm_file_list)))
        ivector_calibrate_list = []
        for idxP, ivector_pdr_file in enumerate(ivector_calibration_file_list):
            ivector_compute_data = {}
            # file_order = int(ivector_pdr.split('/')[-1].split('.')[0].split('_')[1])
            pdr_id = ivector_pdr_file.split('/')[-2]
            ofile = open(ivector_pdr_file, "rb")
            ivector_pdr = dill.load(ofile)
            ofile.close()
            ivector_pdr = SpheModel.transform(ivector_pdr)
            ivector_pdr = LDAModel.transform(ivector_pdr.reshape(1,-1))
            ivector_pdr = PLDAModel.model.transform(ivector_pdr, from_space='D', to_space='U_model')
            listScoreBSV = []
            listBSVFileComp = []
            matchSpeaker = []
            for idxQ, ivector_qst_file in enumerate(ivector_calibration_file_list):
                # file_order = int(ivector_qst.split('/')[-1].split('.')[0].split('_')[1])
                if (ivector_pdr_file == ivector_qst_file):
                    continue
                qst_id = ivector_qst_file.split('/')[-2]
                ofile = open(ivector_qst_file, "rb")
                ivector_qst = dill.load(ofile)
                ofile.close()
                ivector_qst = SpheModel.transform(ivector_qst)
                ivector_qst = LDAModel.transform(ivector_qst.reshape(1,-1))
                ivector_qst = PLDAModel.model.transform(ivector_qst, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_pdr, ivector_qst) 
                listScoreBSV.append(mtxScore)
                listBSVFileComp.append(qst_id)
                matchSpeaker.append(int((pdr_id == qst_id)))
                print("ivector QST n {:} de {:}".format(idxQ,len(ivector_calibration_file_list)-1))
                
            listScoreWSV = []
            listWSVFileComp = []
            for idxU, ivector_ubm_file in enumerate(ivector_ubm_file_list):
                # file_order = int(ivector_ubm.split('/')[-1].split('.')[0].split('_')[1])
                ubm_id = ivector_ubm_file.split('/')[-2]
                ofile = open(ivector_ubm_file, "rb")
                ivector_ubm = dill.load(ofile)
                ofile.close()   
                ivector_ubm = SpheModel.transform(ivector_ubm)
                ivector_ubm = LDAModel.transform(ivector_ubm.reshape(1,-1))
                ivector_ubm = PLDAModel.model.transform(ivector_ubm, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_pdr, ivector_ubm) 
                listScoreWSV.append(mtxScore)
                listWSVFileComp.append(ubm_id)
                print("ivector UBM n {:} de {:}".format(idxU,len(ivector_ubm_file_list)-1))
                
            ivector_compute_data["Speaker_id"] = pdr_id
            ivector_compute_data["speaker_BSV"] = listScoreBSV
            ivector_compute_data["speaker_WSV"] = listScoreWSV
            ivector_compute_data["listBSVFileComp"] = listBSVFileComp
            ivector_compute_data["listWSVFileComp"] = listWSVFileComp
            ivector_compute_data["matchSpeaker"] = matchSpeaker
            ivector_calibrate_list.append(ivector_compute_data)
            print("Fim de arquivo {:} de {:}",idxP,len(ivector_calibration_file_list))
        ofile = open(c.IVECTOR_CALIBRATE_FILEDATA, "wb")
        dill.dump(ivector_calibrate_list, ofile)
        ofile.close()
    # =========================================================================
    if (LDA_IVECTOR_VALIDATE_RUN):
        # --- usa base de validacao
        ivector_validation_file_list = list_contend(c.IVECTOR_MODEL_VALIDATION_DIR,pattern)
        ivector_validation_file_list.sort()
        
        print('Etapa de validacao - ivector')
        print('Arquivos de ivector Valida: {:}'.format(len(ivector_validation_file_list)))
        print('Arquivos de ivector BSV: {:}'.format(len(ivector_calibration_file_list)))
        print('Arquivos de ivector WSV: {:}'.format(len(ivector_ubm_file_list)))
        
        list_pair_validade = []    
        for idxP, ivector_pdr_file in enumerate(ivector_validation_file_list):
            order_pdr = int(ivector_pdr_file.split('/')[-1].split('.')[0].split('_')[1])
            pdr_id = ivector_pdr_file.split('/')[-2]
            for idxQ, ivector_qst_file in enumerate(ivector_validation_file_list):
                order_qst = int(ivector_qst_file.split('/')[-1].split('.')[0].split('_')[1])
                qst_id = ivector_qst_file.split('/')[-2]
                if (pdr_id == qst_id) and (not (order_pdr == order_qst)):
                    valid_pair = {}
                    valid_pair['pdr'] = ivector_pdr_file
                    valid_pair['qst'] = ivector_qst_file
                    list_pair_validade.append(valid_pair)
                    break;
            print("ivector PADRAO n {:} de {:}".format(idxP,len(ivector_validation_file_list)-1))
                    
        ivector_validate_list = []
        for idxD, dict_ivector in enumerate(list_pair_validade):
            ivector_compute_data = {}
            pair_id = dict_ivector['pdr'].split('/')[-2]
            ofile = open(dict_ivector['pdr'], "rb")
            ivector_00 = dill.load(ofile)
            ofile.close()
            ofile = open(dict_ivector['qst'], "rb")
            ivector_01 = dill.load(ofile)
            ofile.close()
            
            ivector_00 = SpheModel.transform(ivector_00)
            ivector_00 = LDAModel.transform(ivector_00.reshape(1,-1))
            ivector_00 = PLDAModel.model.transform(ivector_00, from_space='D', to_space='U_model')
            
            ivector_01 = SpheModel.transform(ivector_01)
            ivector_01 = LDAModel.transform(ivector_01.reshape(1,-1))
            ivector_01 = PLDAModel.model.transform(ivector_01, from_space='D', to_space='U_model')
            
            pairScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_00, ivector_01) 
            listScoreBSV = []
            listBSVFileComp = []
            for idxQ, ivector_qst_file in enumerate(ivector_calibration_file_list):
                qst_id = ivector_qst_file.split('/')[-2]
                ofile = open(ivector_qst_file, "rb")
                ivector_qst = dill.load(ofile)
                ofile.close()
                ivector_qst = SpheModel.transform(ivector_qst)
                ivector_qst = LDAModel.transform(ivector_qst.reshape(1,-1))
                ivector_qst = PLDAModel.model.transform(ivector_qst, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_01, ivector_qst) 
                listScoreBSV.append(mtxScore)
                listBSVFileComp.append(qst_id)
             
            listScoreWSV = []
            listWSVFileComp = []
            for idxU, ivector_ubm_file in enumerate(ivector_ubm_file_list):
                ubm_id = ivector_ubm_file.split('/')[-2]
                ofile = open(ivector_ubm_file, "rb")
                ivector_ubm = dill.load(ofile)
                ofile.close()   
                ivector_ubm = SpheModel.transform(ivector_ubm)
                ivector_ubm = LDAModel.transform(ivector_ubm.reshape(1,-1))
                ivector_ubm = PLDAModel.model.transform(ivector_ubm, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_01, ivector_ubm) 
                listScoreWSV.append(mtxScore)
                listWSVFileComp.append(ubm_id)
                
                
            ivector_compute_data["Speaker_id"] = pair_id
            ivector_compute_data["Pair_Score"] = pairScore
            ivector_compute_data["speaker_BSV"] = listScoreBSV
            ivector_compute_data["speaker_WSV"] = listScoreWSV
            ivector_compute_data["listBSVFileComp"] = listBSVFileComp
            ivector_compute_data["listWSVFileComp"] = listWSVFileComp
            ivector_validate_list.append(ivector_compute_data)
            print("Fim de arquivo {:} de {:}",idxD,len(list_pair_validade))
        ofile = open(c.IVECTOR_VALIDATE_FILEDATA, "wb")
        dill.dump(ivector_validate_list, ofile)
        ofile.close()
        
# =============================================================================
if (LDA_XVECTOR_TDDNN_RUN):
    if (not os.path.exists(c.XVECTOR_TDDNN_SPH_LDA_PLDA_MODELS)):
        model = {}
        # --- usa base LDA ----------------------------------------------------
        lda_ivectos_file_list = list_contend(c.XVECTOR_TDDNN_MODEL_LDA_DIR, pattern)
        nFiles = len(lda_ivectos_file_list)
        y = np.zeros((nFiles,))    
        for idx, file_name in enumerate(lda_ivectos_file_list):
            speaker_id = file_name.split('/')[-2]
            ofile = open(file_name, "rb")
            spk_vector = dill.load(ofile)
            ofile.close()
            y[idx] = int(speaker_id);
            if (idx == 0):
                embedding_size = spk_vector.shape[0]
                X = np.zeros((nFiles,embedding_size))
            X[idx,:] = np.array(spk_vector)
        # --- processo de Sphearing -------------------------------------------
        SpheModel = Sphering.SpheringSVD()
        SpheModel.fit(X)
        Xsph = SpheModel.transform(X)
        LDAModel = LinearDiscriminantAnalysis()
        LDAModel.fit(Xsph, y)
        PLDAModel = plda.Classifier()
        Xlda = LDAModel.transform(Xsph)
        PLDAModel.fit_model(Xlda, y)
        model['sph'] = SpheModel
        model['lda'] = LDAModel
        model['plda'] = PLDAModel
        ofile = open(c.XVECTOR_TDDNN_SPH_LDA_PLDA_MODELS, "wb")
        dill.dump(model, ofile)
        ofile.close()
    else:
        ofile = open(c.XVECTOR_TDDNN_SPH_LDA_PLDA_MODELS, "rb")
        model = dill.load(ofile)
        ofile.close()
        SpheModel = model['sph']
        LDAModel = model['lda']
        PLDAModel = model['plda']
    # =========================================================================
    if (LDA_XVECTOR_TDDNN_CALIBRATE_RUN):
        # --- usa base de calibracao
        ivector_calibration_file_list = list_contend(c.XVECTOR_TDDNN_MODEL_CALIBRATION_DIR,pattern)
        ivector_calibration_file_list.sort()    
        ivector_ubm_file_list = list_contend(c.XVECTOR_TDDNN_MODEL_UBM_DIR,pattern)
        ivector_ubm_file_list.sort()
        print('Etapa de calibracao - ivector')
        print('Arquivos de ivector BSV: {:}'.format(len(ivector_calibration_file_list)))
        print('Arquivos de ivector WSV: {:}'.format(len(ivector_ubm_file_list)))
        ivector_calibrate_list = []
        for idxP, ivector_pdr_file in enumerate(ivector_calibration_file_list):
            ivector_compute_data = {}
            # file_order = int(ivector_pdr.split('/')[-1].split('.')[0].split('_')[1])
            pdr_id = ivector_pdr_file.split('/')[-2]
            ofile = open(ivector_pdr_file, "rb")
            ivector_pdr = dill.load(ofile)
            ofile.close()
            ivector_pdr = SpheModel.transform(ivector_pdr)
            ivector_pdr = LDAModel.transform(ivector_pdr.reshape(1,-1))
            ivector_pdr = PLDAModel.model.transform(ivector_pdr, from_space='D', to_space='U_model')
            listScoreBSV = []
            listBSVFileComp = []
            matchSpeaker = []
            for idxQ, ivector_qst_file in enumerate(ivector_calibration_file_list):
                # file_order = int(ivector_qst.split('/')[-1].split('.')[0].split('_')[1])
                if (ivector_pdr_file == ivector_qst_file):
                    continue
                qst_id = ivector_qst_file.split('/')[-2]
                ofile = open(ivector_qst_file, "rb")
                ivector_qst = dill.load(ofile)
                ofile.close()
                ivector_qst = SpheModel.transform(ivector_qst)
                ivector_qst = LDAModel.transform(ivector_qst.reshape(1,-1))
                ivector_qst = PLDAModel.model.transform(ivector_qst, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_pdr, ivector_qst) 
                listScoreBSV.append(mtxScore)
                listBSVFileComp.append(qst_id)
                matchSpeaker.append(int((pdr_id == qst_id)))
             
            listScoreWSV = []
            listWSVFileComp = []
            for idxU, ivector_ubm in enumerate(ivector_ubm_file_list):
                # file_order = int(ivector_ubm.split('/')[-1].split('.')[0].split('_')[1])
                ubm_id = ivector_ubm.split('/')[-2]
                ofile = open(ivector_ubm, "rb")
                ivector_ubm = dill.load(ofile)
                ofile.close()   
                ivector_ubm = SpheModel.transform(ivector_ubm)
                ivector_ubm = LDAModel.transform(ivector_ubm.reshape(1,-1))
                ivector_ubm = PLDAModel.model.transform(ivector_ubm, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_pdr, ivector_ubm) 
                listScoreWSV.append(mtxScore)
                listWSVFileComp.append(ubm_id)
                
            ivector_compute_data["Speaker_id"] = pdr_id
            ivector_compute_data["speaker_BSV"] = listScoreBSV
            ivector_compute_data["speaker_WSV"] = listScoreWSV
            ivector_compute_data["listBSVFileComp"] = listBSVFileComp
            ivector_compute_data["listWSVFileComp"] = listWSVFileComp
            ivector_compute_data["matchSpeaker"] = matchSpeaker
            ivector_calibrate_list.append(ivector_compute_data)
            print("Fim de arquivo {:} de {:}",idxP,len(ivector_calibration_file_list))
        ofile = open(c.XVECTOR_TDDNN_CALIBRATE_FILEDATA, "wb")
        dill.dump(ivector_calibrate_list, ofile)
        ofile.close()
    # =========================================================================
    if (LDA_XVECTOR_TDDNN_VALIDATE_RUN):
        # --- usa base de validacao
        ivector_validation_file_list = list_contend(c.XVECTOR_TDDNN_MODEL_VALIDATION_DIR,pattern)
        ivector_validation_file_list.sort()
        
        print('Etapa de validacao - ivector')
        print('Arquivos de ivector Valida: {:}'.format(len(ivector_validation_file_list)))
        print('Arquivos de ivector BSV: {:}'.format(len(ivector_calibration_file_list)))
        print('Arquivos de ivector WSV: {:}'.format(len(ivector_ubm_file_list)))
        
        list_pair_validade = []    
        for idxP, ivector_pdr in enumerate(ivector_validation_file_list):
            order_pdr = int(ivector_pdr.split('/')[-1].split('.')[0].split('_')[1])
            pdr_id = ivector_pdr.split('/')[-2]
            for idxQ, ivector_qst in enumerate(ivector_validation_file_list):
                order_qst = int(ivector_qst.split('/')[-1].split('.')[0].split('_')[1])
                qst_id = ivector_qst.split('/')[-2]
                if (pdr_id == qst_id) and (not (order_pdr == order_qst)):
                    valid_pair = {}
                    valid_pair['pdr'] = ivector_pdr
                    valid_pair['qst'] = ivector_qst
                    list_pair_validade.append(valid_pair)
                    break;
                    
        ivector_validate_list = []
        for idxD, dict_ivector in enumerate(list_pair_validade):
            ivector_compute_data = {}
            pair_id = dict_ivector['pdr'].split('/')[-2]
            ofile = open(dict_ivector['pdr'], "rb")
            ivector_00 = dill.load(ofile)
            ofile.close()
            ofile = open(dict_ivector['qst'], "rb")
            ivector_01 = dill.load(ofile)
            ofile.close()
            
            ivector_00 = SpheModel.transform(ivector_00)
            ivector_00 = LDAModel.transform(ivector_00.reshape(1,-1))
            ivector_00 = PLDAModel.model.transform(ivector_00, from_space='D', to_space='U_model')
            
            ivector_01 = SpheModel.transform(ivector_01)
            ivector_01 = LDAModel.transform(ivector_01.reshape(1,-1))
            ivector_01 = PLDAModel.model.transform(ivector_01, from_space='D', to_space='U_model')
            
            pairScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_00, ivector_01) 
            listScoreBSV = []
            listBSVFileComp = []
            for idxQ, ivector_qst in enumerate(ivector_calibration_file_list):
                qst_id = ivector_qst.split('/')[-2]
                ofile = open(ivector_qst, "rb")
                ivector_qst = dill.load(ofile)
                ofile.close()
                ivector_qst = SpheModel.transform(ivector_qst)
                ivector_qst = LDAModel.transform(ivector_qst.reshape(1,-1))
                ivector_qst = PLDAModel.model.transform(ivector_qst, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_01, ivector_qst) 
                listScoreBSV.append(mtxScore)
                listBSVFileComp.append(qst_id)
             
            listScoreWSV = []
            listWSVFileComp = []
            for idxU, ivector_ubm in enumerate(ivector_ubm_file_list):
                ubm_id = ivector_ubm.split('/')[-2]
                ofile = open(ivector_ubm, "rb")
                ivector_ubm = dill.load(ofile)
                ofile.close()   
                ivector_ubm = SpheModel.transform(ivector_ubm)
                ivector_ubm = LDAModel.transform(ivector_ubm.reshape(1,-1))
                ivector_ubm = PLDAModel.model.transform(ivector_ubm, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_01, ivector_ubm) 
                listScoreWSV.append(mtxScore)
                listWSVFileComp.append(ubm_id)
                
            ivector_compute_data["Speaker_id"] = pair_id
            ivector_compute_data["Pair_Score"] = pairScore
            ivector_compute_data["speaker_BSV"] = listScoreBSV
            ivector_compute_data["speaker_WSV"] = listScoreWSV
            ivector_compute_data["listBSVFileComp"] = listBSVFileComp
            ivector_compute_data["listWSVFileComp"] = listWSVFileComp
            ivector_validate_list.append(ivector_compute_data)
            print("Fim de arquivo {:} de {:}",idxD,len(list_pair_validade))
        ofile = open(c.XVECTOR_TDDNN_VALIDATE_FILEDATA, "wb")
        dill.dump(ivector_validate_list, ofile)
        ofile.close()

# =============================================================================
if (LDA_XVECTOR_RESNET_RUN):
    if (not os.path.exists(c.XVECTOR_RESNET_SPH_LDA_PLDA_MODELS)):
        model = {}
        # --- usa base LDA ----------------------------------------------------
        lda_ivectos_file_list = list_contend(c.XVECTOR_RESNET_MODEL_LDA_DIR, pattern)
        nFiles = len(lda_ivectos_file_list)
        y = np.zeros((nFiles,))    
        for idx, file_name in enumerate(lda_ivectos_file_list):
            speaker_id = file_name.split('/')[-2]
            ofile = open(file_name, "rb")
            spk_vector = dill.load(ofile)
            ofile.close()
            y[idx] = int(speaker_id);
            if (idx == 0):
                embedding_size = spk_vector.shape[0]
                X = np.zeros((nFiles,embedding_size))
            X[idx,:] = np.array(spk_vector)
        # --- processo de Sphearing -------------------------------------------
        SpheModel = Sphering.SpheringSVD()
        SpheModel.fit(X)
        Xsph = SpheModel.transform(X)
        LDAModel = LinearDiscriminantAnalysis()
        LDAModel.fit(Xsph, y)
        PLDAModel = plda.Classifier()
        Xlda = LDAModel.transform(Xsph)
        PLDAModel.fit_model(Xlda, y)
        model['sph'] = SpheModel
        model['lda'] = LDAModel
        model['plda'] = PLDAModel
        ofile = open(c.XVECTOR_RESNET_SPH_LDA_PLDA_MODELS, "wb")
        dill.dump(model, ofile)
        ofile.close()
    else:
        ofile = open(c.XVECTOR_RESNET_SPH_LDA_PLDA_MODELS, "rb")
        model = dill.load(ofile)
        ofile.close()
        SpheModel = model['sph']
        LDAModel = model['lda']
        PLDAModel = model['plda']
    # =========================================================================
    if (LDA_XVECTOR_RESNET_CALIBRATE_RUN):
        # --- usa base de calibracao
        ivector_calibration_file_list = list_contend(c.XVECTOR_RESNET_MODEL_CALIBRATION_DIR,pattern)
        ivector_calibration_file_list.sort()    
        ivector_ubm_file_list = list_contend(c.XVECTOR_RESNET_MODEL_UBM_DIR,pattern)
        ivector_ubm_file_list.sort()
        print('Etapa de calibracao - ivector')
        print('Arquivos de ivector BSV: {:}'.format(len(ivector_calibration_file_list)))
        print('Arquivos de ivector WSV: {:}'.format(len(ivector_ubm_file_list)))
        ivector_calibrate_list = []
        for idxP, ivector_pdr_file in enumerate(ivector_calibration_file_list):
            ivector_compute_data = {}
            # file_order = int(ivector_pdr.split('/')[-1].split('.')[0].split('_')[1])
            pdr_id = ivector_pdr_file.split('/')[-2]
            ofile = open(ivector_pdr_file, "rb")
            ivector_pdr = dill.load(ofile)
            ofile.close()
            ivector_pdr = SpheModel.transform(ivector_pdr)
            ivector_pdr = LDAModel.transform(ivector_pdr.reshape(1,-1))
            ivector_pdr = PLDAModel.model.transform(ivector_pdr, from_space='D', to_space='U_model')
            listScoreBSV = []
            listBSVFileComp = []
            matchSpeaker = []
            for idxQ, ivector_qst_file in enumerate(ivector_calibration_file_list):
                # file_order = int(ivector_qst.split('/')[-1].split('.')[0].split('_')[1])
                if (ivector_pdr_file == ivector_qst_file):
                    continue
                qst_id = ivector_qst_file.split('/')[-2]
                ofile = open(ivector_qst_file, "rb")
                ivector_qst = dill.load(ofile)
                ofile.close()
                ivector_qst = SpheModel.transform(ivector_qst)
                ivector_qst = LDAModel.transform(ivector_qst.reshape(1,-1))
                ivector_qst = PLDAModel.model.transform(ivector_qst, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_pdr, ivector_qst) 
                listScoreBSV.append(mtxScore)
                listBSVFileComp.append(qst_id)
                matchSpeaker.append(int((pdr_id == qst_id)))
             
            listScoreWSV = []
            listWSVFileComp = []
            for idxU, ivector_ubm in enumerate(ivector_ubm_file_list):
                # file_order = int(ivector_ubm.split('/')[-1].split('.')[0].split('_')[1])
                ubm_id = ivector_ubm.split('/')[-2]
                ofile = open(ivector_ubm, "rb")
                ivector_ubm = dill.load(ofile)
                ofile.close()   
                ivector_ubm = SpheModel.transform(ivector_ubm)
                ivector_ubm = LDAModel.transform(ivector_ubm.reshape(1,-1))
                ivector_ubm = PLDAModel.model.transform(ivector_ubm, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_pdr, ivector_ubm) 
                listScoreWSV.append(mtxScore)
                listWSVFileComp.append(ubm_id)
                
            ivector_compute_data["Speaker_id"] = pdr_id
            ivector_compute_data["speaker_BSV"] = listScoreBSV
            ivector_compute_data["speaker_WSV"] = listScoreWSV
            ivector_compute_data["listBSVFileComp"] = listBSVFileComp
            ivector_compute_data["listWSVFileComp"] = listWSVFileComp
            ivector_compute_data["matchSpeaker"] = matchSpeaker
            ivector_calibrate_list.append(ivector_compute_data)
            print("Fim de arquivo {:} de {:}",idxP,len(ivector_calibration_file_list))    
        ofile = open(c.XVECTOR_RESNET_CALIBRATE_FILEDATA, "wb")
        dill.dump(ivector_calibrate_list, ofile)
        ofile.close()
    # =========================================================================
    if (LDA_XVECTOR_RESNET_VALIDATE_RUN):
        # --- usa base de validacao
        ivector_validation_file_list = list_contend(c.XVECTOR_RESNET_MODEL_VALIDATION_DIR,pattern)
        ivector_validation_file_list.sort()
        
        print('Etapa de validacao - ivector')
        print('Arquivos de ivector Valida: {:}'.format(len(ivector_validation_file_list)))
        print('Arquivos de ivector BSV: {:}'.format(len(ivector_calibration_file_list)))
        print('Arquivos de ivector WSV: {:}'.format(len(ivector_ubm_file_list)))
        
        list_pair_validade = []    
        for idxP, ivector_pdr in enumerate(ivector_validation_file_list):
            order_pdr = int(ivector_pdr.split('/')[-1].split('.')[0].split('_')[1])
            pdr_id = ivector_pdr.split('/')[-2]
            for idxQ, ivector_qst in enumerate(ivector_validation_file_list):
                order_qst = int(ivector_qst.split('/')[-1].split('.')[0].split('_')[1])
                qst_id = ivector_qst.split('/')[-2]
                if (pdr_id == qst_id) and (not (order_pdr == order_qst)):
                    valid_pair = {}
                    valid_pair['pdr'] = ivector_pdr
                    valid_pair['qst'] = ivector_qst
                    list_pair_validade.append(valid_pair)
                    break;
                    
        ivector_validate_list = []
        for idxD, dict_ivector in enumerate(list_pair_validade):
            ivector_compute_data = {}
            pair_id = dict_ivector['pdr'].split('/')[-2]
            ofile = open(dict_ivector['pdr'], "rb")
            ivector_00 = dill.load(ofile)
            ofile.close()
            ofile = open(dict_ivector['qst'], "rb")
            ivector_01 = dill.load(ofile)
            ofile.close()
            
            ivector_00 = SpheModel.transform(ivector_00)
            ivector_00 = LDAModel.transform(ivector_00.reshape(1,-1))
            ivector_00 = PLDAModel.model.transform(ivector_00, from_space='D', to_space='U_model')
            
            ivector_01 = SpheModel.transform(ivector_01)
            ivector_01 = LDAModel.transform(ivector_01.reshape(1,-1))
            ivector_01 = PLDAModel.model.transform(ivector_01, from_space='D', to_space='U_model')
            
            pairScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_00, ivector_01) 
            listScoreBSV = []
            listBSVFileComp = []
            for idxQ, ivector_qst in enumerate(ivector_calibration_file_list):
                qst_id = ivector_qst.split('/')[-2]
                ofile = open(ivector_qst, "rb")
                ivector_qst = dill.load(ofile)
                ofile.close()
                ivector_qst = SpheModel.transform(ivector_qst)
                ivector_qst = LDAModel.transform(ivector_qst.reshape(1,-1))
                ivector_qst = PLDAModel.model.transform(ivector_qst, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_01, ivector_qst) 
                listScoreBSV.append(mtxScore)
                listBSVFileComp.append(qst_id)
             
            listScoreWSV = []
            listWSVFileComp = []
            for idxU, ivector_ubm in enumerate(ivector_ubm_file_list):
                ubm_id = ivector_ubm.split('/')[-2]
                ofile = open(ivector_ubm, "rb")
                ivector_ubm = dill.load(ofile)
                ofile.close()   
                ivector_ubm = SpheModel.transform(ivector_ubm)
                ivector_ubm = LDAModel.transform(ivector_ubm.reshape(1,-1))
                ivector_ubm = PLDAModel.model.transform(ivector_ubm, from_space='D', to_space='U_model')
                mtxScore = PLDAModel.model.calc_same_diff_log_likelihood_ratio(ivector_01, ivector_ubm) 
                listScoreWSV.append(mtxScore)
                listWSVFileComp.append(ubm_id)
                
            ivector_compute_data["Speaker_id"] = pair_id
            ivector_compute_data["Pair_Score"] = pairScore
            ivector_compute_data["speaker_BSV"] = listScoreBSV
            ivector_compute_data["speaker_WSV"] = listScoreWSV
            ivector_compute_data["listBSVFileComp"] = listBSVFileComp
            ivector_compute_data["listWSVFileComp"] = listWSVFileComp
            ivector_validate_list.append(ivector_compute_data)
            print("Fim de arquivo {:} de {:}",idxD,len(list_pair_validade))
        ofile = open(c.XVECTOR_RESNET_VALIDATE_FILEDATA, "wb")
        dill.dump(ivector_validate_list, ofile)
        ofile.close()