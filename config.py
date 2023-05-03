#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:59:30 2022

@author: adelino
"""
# Arquivo de configuracoes

# Experimentos, testes e calibracao

CURRENT_RAW_FEATURE_FILE_LIST = []

# extention file for raw and normalized features
RAW_FEAT_EXT = ".rfeat"
NRM_FEAT_EXT = ".nfeat"

# TODO:  use extention pattern
PY_MODEL_EXT = ".pthmdl"
PY_DB_EXT = ".pthdb"
PY_DATA_EXT = ".pthdt"
PY_LIST_EXT = ".pthlst"


FULL_FEATURE_LIST = ["spectogram","time_domain","freq_domain","spc_entropy",
                     "LTAS","vad_sohn","S2NR","pitch","formants","mfcc","pncc",
                     "plp","rasta_plp","ssch", "zcpa","teocc","mfec"]
ENVIROMENT="DEV"
PATTERN_AUDIO_EXT = ('.3gp','.aa','.aac','.aax','.act','.aiff','.amr','.ape','.au',
           '.awb','.dct','.dss','.dvf','.flac','.gsm','.iklax','.ivs',
           '.m4a','.m4b','.m4p','.mmf','.mp3','.mpc','.msv','.nmf','.nsf',
           '.ogg','.oga','.mogg','.opus','.ra','.rm','.raw','.sln','.tta',
           '.vox','.wav','.wma','.wv','.webm','.8svx')

SPEAKER_SETS = ["LDA","UBM","TDDNN","RESNET","CALIBRATION","VALIDATION"]
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
# TODO: change file extention of files to 
FEATURE_FILE_LIST = '../BaseDadosFeatures/dbFeatureFileList.pthlst'
FEATURE_FILE_OUTPUT = '../BaseDadosFeatures/'
DB_TIME_INFO_FILE = '../BaseDadosFeatures/dbFeatureVadTimeInfo.pthdb'
DB_TIME_SEPR_FILE = '../BaseDadosFeatures/dbTrainingGroups.pthdb' 
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
UBM_FILE_NAME = '../BaseDadosNormalizada/ubmMeanStDevFile.pthdt'
GMM_UBM_DATA_FILE_NAME = '../BaseDadosNormalizada/ubmGMMData.pthdt'
GMM_UBM_FILE_NAME = '../BaseDadosNormalizada/ubmGMMmodel.pthmdl'
UBM_covType='diag'
UBM_nComponents = 512
# ==============================================================================
TDDNN_SAVE_MODELS_DIR = "../BaseDNNTrained/TDDNN/"
TDDNN_SAVE_MODELS_FILE_BASE = "model_tddnn"
TDDNN_TRAIN_PATH = '../BaseDadosNormalizada/TDDNN/'
TDDNN_NUM_WIN_SIZE = 200 
TDDNN_TRAIN_TEST_RATIO = 15
TDDNN_N_EPOCHS = 1750
TDDNN_LR = 1e-1 # Initial learning rate
TDDNN_WD = 1e-4 # Weight decay (L2 penalty)
TDDNN_OPT_TYPE = 'ASGD' # ex) sgd, adam, adagrad, LBFGS (muita memoria), ASGD
TDDNN_BATCH_SIZE = 16 # Batch size for training # original 64
TDDNN_VALID_BATCH = 2 # Batch size for validation
TDDNN_USE_SHUFFLE = True
TDDNN_TRAIN_DATA_FILE = 'tddnn_train_data.txt'
# ==============================================================================
ALPHA_LNORM = 10
# ==============================================================================
RESNET_SAVE_MODELS_DIR = "../BaseDNNTrained/RESNET/"
RESNET_SAVE_MODELS_FILE_BASE = "model_resnet"
RESNET_TRAIN_PATH = '../BaseDadosNormalizada/TDDNN/'
RESNET_NUM_WIN_SIZE = 200 
RESNET_TRAIN_TEST_RATIO = 15
RESNET_N_EPOCHS = 1750
RESNET_LR = 1e-1 # Initial learning rate
RESNET_WD = 1e-4 # Weight decay (L2 penalty)
RESNET_OPT_TYPE = 'ASGD' # ex) sgd, adam, adagrad, LBFGS (muita memoria), ASGD
RESNET_BATCH_SIZE = 16 # Batch size for training # original 64
RESNET_VALID_BATCH = 2 # Batch size for validation # original 8
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
IVECTOR_SPH_LDA_PLDA_MODELS = '../BaseModels/ivector_sph_lda_plda_models.pth'

XVECTOR_TDDNN_MODEL_CALIBRATION_DIR = '../BaseModels/XVECTOR_CALIBRATION/TDDNN/'
XVECTOR_TDDNN_MODEL_VALIDATION_DIR = '../BaseModels/XVECTOR_VALIDATION/TDDNN/'
XVECTOR_TDDNN_MODEL_LDA_DIR = '../BaseModels/XVECTOR_LDA/TDDNN/'
XVECTOR_TDDNN_MODEL_UBM_DIR = '../BaseModels/XVECTOR_UBM/TDDNN/'
XVECTOR_TDDNN_SPH_LDA_PLDA_MODELS = '../BaseModels/xvector_tddnn_sph_lda_plda_models.pth'

XVECTOR_RESNET_MODEL_CALIBRATION_DIR = '../BaseModels/XVECTOR_CALIBRATION/RESNET/'
XVECTOR_RESNET_MODEL_VALIDATION_DIR = '../BaseModels/XVECTOR_VALIDATION/RESNET/'
XVECTOR_RESNET_MODEL_LDA_DIR = '../BaseModels/XVECTOR_LDA/RESNET/'
XVECTOR_RESNET_MODEL_UBM_DIR = '../BaseModels/XVECTOR_UBM/RESNET/'
XVECTOR_RESNET_SPH_LDA_PLDA_MODELS = '../BaseModels/xvector_resnet_sph_lda_plda_models.pth'
# =============================================================================
T_MATRIX_MAX_ITER = 200
T_MATRIX_DIM = 600
T_MATRIX_STATS_TRAIN_FILE_IVECTOR = '../BaseDadosNormalizada/statsTrainFormTmatrixIvector.p'
T_MATRIX_FILE_IVECTOR = '../BaseDadosNormalizada/TmatrixFromIvector.p'
# =============================================================================
GMM_CALIBRATE_RESULTS = "../Resultados/UBM_GMM/"
GMM_CALIBRATE_FILEDATA = "../Resultados/UBM_GMM/gmm_calibrate_file_data.pth"
GMM_VALIDATE_FILEDATA = "../Resultados/UBM_GMM/gmm_validate_file_data.pth"
GMM_RESULT_PARAM = "../Resultados/UBM_GMM/gmm_result_param.pth"
GMM_VALIDATE_RESULT = "../Resultados/UBM_GMM/gmm_validade_result.pth"

# =============================================================================
IVECTOR_CALIBRATE_FILEDATA = '../Resultados/IVECTOR/ivector_calibrate_file_data.pth'
IVECTOR_VALIDATE_FILEDATA = '../Resultados/IVECTOR/ivector_validate_file_data.pth'
IVECTOR_RESULT_PARAM = '../Resultados/IVECTOR/ivector_result_param.pth'
IVECTOR_VALIDATE_RESULT = '../Resultados/IVECTOR/ivector_validate_result.pth'

XVECTOR_TDDNN_CALIBRATE_FILEDATA = '../Resultados/XVECTOR/TDDNN/xvector_tddnn_calibrate_file_data.pth'
XVECTOR_TDDNN_VALIDATE_FILEDATA = '../Resultados/XVECTOR/TDDNN/xvector_tddnn_validate_file_data.pth'
XVECTOR_TDDNN_RESULT_PARAM = '../Resultados/XVECTOR/TDDNN/xvector_tddnn_result_param.pth'
XVECTOR_TDDNN_VALIDATE_RESULT= '../Resultados/XVECTOR/TDDNN/xvector_tddnn_validate_result.pth'

XVECTOR_RESNET_CALIBRATE_FILEDATA = '../Resultados/XVECTOR/RESNET/xvector_resnet_calibrate_file_data.pth'
XVECTOR_RESNET_VALIDATE_FILEDATA = '../Resultados/XVECTOR/RESNET/xvector_resnet_validate_file_data.pth'
XVECTOR_RESNET_RESULT_PARAM = '../Resultados/XVECTOR/RESNET/xvector_resnet_result_param.pth'
XVECTOR_RESNET_VALIDATE_RESULT = '../Resultados/XVECTOR/RESNET/xvector_resnet_validate_result.pth'


# ==============================================================================
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
