#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:59:30 2022

@author: adelino
"""
# Arquivo de configuracoes

# Experimentos, testes e calibracao


# ------------------------------------------------------------------------------
SAMPLE_RATE = 8000
CHANNELS = '1'
CODEC = 'PCM'
FILE_TYPE = 'WAV'
# ------------------------------------------------------------------------------
WIN_SIZE = 0.025
STEP_SIZE = 0.01
# ------------------------------------------------------------------------------
# Etapa 1 calculo de caracteristicas pragmaticas
# Audios de entrada
AUDIO_FILE_INPUT = '../BaseDadosAudios/'
# Caracteristicas de saida
FEATURE_FILE_LIST = '../BaseDadosFeatures/dbFeatureFileList.txt'
FEATURE_FILE_OUTPUT = '../BaseDadosFeatures/'
DB_TIME_INFO_FILE = '../BaseDadosFeatures/dbFeatureVadTimeInfo.txt'
DB_TIME_SEPR_FILE = '../BaseDadosFeatures/dbTrainingGroups.txt'
# ==============================================================================
'''
A separação em conjunto de treinamento, teste e validação será realizado depois 
do calculo das caracteristicas

AUDIO_TRAIN_PATH = '../BaseDadosAudios/Treinamento/'
AUDIO_TESTI_PATH = '../BaseDadosAudios/Teste/'
AUDIO_VALID_PATH = '../BaseDadosAudios/Validacao/'
# Caracteristicas calculadas
FEATURES_TRAIN_PATH = '../BaseDadosFeatures/Treinamento/'
FEATURES_TESTI_PATH = '../BaseDadosFeatures/Teste/'
FEATURES_VALID_PATH = '../BaseDadosFeatures/Validacao/'
'''
# ==============================================================================
DB_POOL_EXPERIMENT = "../BaseDadosNormalizada/"
DB_POOL_MIN_TIME = 20
# ==============================================================================
UBM_FILE_NAME = '../BaseDadosNormalizada/ubmMeanStDevFile.txt'
GMM_UBM_DATA_FILE_NAME = '../BaseDadosNormalizada/ubmGMMData.p'
GMM_UBM_FILE_NAME = '../BaseDadosNormalizada/ubmGMMmodel.p'
UBM_covType='diag'
UBM_nComponents = 512
# ==============================================================================
TDDNN_SAVE_MODELS_DIR = "../BaseDNNTrained/TDDNN/"
TDDNN_SAVE_MODELS_FILE_BASE = "model_tddnn"
TDDNN_TRAIN_PATH = '../BaseDadosNormalizada/TDDNN/'
TDDNN_NUM_WIN_SIZE = 200 
TDDNN_TRAIN_TEST_RATIO = 15
TDDNN_N_EPOCHS = 750
TDDNN_LR = 1e-1 # Initial learning rate
TDDNN_WD = 1e-4 # Weight decay (L2 penalty)
TDDNN_OPT_TYPE = 'adam' # ex) sgd, adam, adagrad
TDDNN_BATCH_SIZE = 64 # Batch size for training # original 64
TDDNN_VALID_BATCH = 8 # Batch size for validation
TDDNN_USE_SHUFFLE = True
TDDNN_TRAIN_DATA_FILE = 'tddnn_train_data'
# ==============================================================================
ALPHA_LNORM = 10
# ==============================================================================
RESNET_SAVE_MODELS_DIR = "../BaseDNNTrained/RESNET/"
RESNET_SAVE_MODELS_FILE_BASE = "model_resnet"
RESNET_TRAIN_PATH = '../BaseDadosNormalizada/RESNET/'
RESNET_NUM_WIN_SIZE = 200 
RESNET_TRAIN_TEST_RATIO = 15
RESNET_N_EPOCHS = 750
RESNET_LR = 1e-1 # Initial learning rate
RESNET_WD = 1e-4 # Weight decay (L2 penalty)
RESNET_OPT_TYPE = 'sgd' # ex) sgd, adam, adagrad
RESNET_BATCH_SIZE = 64 # Batch size for training # original 64
RESNET_VALID_BATCH = 8 # Batch size for validation
RESNET_USE_SHUFFLE = True
RESNET_BACKBONETYPE = 'resnet18'
RESNET_TRAIN_DATA_FILE = 'resnet_train_data.txt'
# ==============================================================================
CALIBRATION_DIR = '../BaseDadosNormalizada/CALIBRATION/'
VALIDATION_DIR = '../BaseDadosNormalizada/VALIDATION/'
LDA_DIR = '../BaseDadosNormalizada/LDA/'
UBM_DIR = '../BaseDadosNormalizada/UBM/'
# ==============================================================================
GMM_MODEL_CALIBRATION_DIR = '../BaseModels/GMM_CALIBRATION/'
GMM_MODEL_VALIDATION_DIR = '../BaseModels/GMM_VALIDATION/'
GMM_MODEL_UBM_DIR = '../BaseModels/GMM_UBM/'

IVECTOR_MODEL_CALIBRATION_DIR = '../BaseModels/IVECTOR_CALIBRATION/'
IVECTOR_MODEL_VALIDATION_DIR = '../BaseModels/IVECTOR_VALIDATION/'
IVECTOR_MODEL_LDA_DIR = '../BaseModels/IVECTOR_LDA/'
IVECTOR_MODEL_UBM_DIR = '../BaseModels/IVECTOR_UBM/'

XVECTOR_TDDNN_MODEL_CALIBRATION_DIR = '../BaseModels/XVECTOR_CALIBRATION/TDDNN/'
XVECTOR_TDDNN_MODEL_VALIDATION_DIR = '../BaseModels/XVECTOR_VALIDATION/TDDNN/'
XVECTOR_TDDNN_MODEL_LDA_DIR = '../BaseModels/XVECTOR_LDA/TDDNN/'
XVECTOR_TDDNN_MODEL_UBM_DIR = '../BaseModels/XVECTOR_UBM/TDDNN/'

XVECTOR_RESNET_MODEL_CALIBRATION_DIR = '../BaseModels/XVECTOR_CALIBRATION/RESNET/'
XVECTOR_RESNET_MODEL_VALIDATION_DIR = '../BaseModels/XVECTOR_VALIDATION/RESNET/'
XVECTOR_RESNET_MODEL_LDA_DIR = '../BaseModels/XVECTOR_LDA/RESNET/'
XVECTOR_RESNET_MODEL_UBM_DIR = '../BaseModels/XVECTOR_UBM/RESNET/'
# =============================================================================
T_MATRIX_MAX_ITER = 200
T_MATRIX_DIM = 600
T_MATRIX_STATS_TRAIN_FILE = '../BaseDadosNormalizada/statsTrainFormTmatrixIvector.p'
T_MATRIX_FILE = '../BaseDadosNormalizada/TmatrixFromIvector.p'
# =============================================================================
GMM_CALIBRATE_RESULTS = "../Resultados/UBM_GMM/"
GMM_CALIBRATE_FILEDATA = "../Resultados/UBM_GMM/gmm_calibrate_file_data.pth"
GMM_VALIDATE_FILEDATA = "../Resultados/UBM_GMM/gmm_validate_file_data.pth"
# =============================================================================

# Dados para calculo das características
# --- PRE-ENFASE
PRE_ENPHASIS_COEF = 0.975
# --- FORMANTES
# com base nas caracteriticas do audio e SAMPLE_RATE
NUM_CEPS_COEFS = 13
# --- MFCC
MFCC_NUM_FILTERS = 13
# --- PLP and RASPA-PLP
PLP_NUM_FILTERS = 17
PLP_SUM_POWER = True
PLP_LIFTER_COEF = 0.6
# --- PNCC
PNCC_NUM_FILTERS = 35
# PLP_DITHER = False
# ==============================================================================