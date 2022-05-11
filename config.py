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
# ------------------------------------------------------------------------------
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
DB_POOL_EXPERIMENT = "../BaseDadosNormalizada/"
DB_POOL_MIN_TIME = 10
# ------------------------------------------------------------------------------
UBM_FILE_NAME = '../BaseDadosNormalizada/ubmMeanStDevFile.txt'
GMM_UBM_DATA_FILE_NAME = '../BaseDadosNormalizada/ubmGMMData.p'
GMM_UBM_FILE_NAME = '../BaseDadosNormalizada/ubmGMMmodel.p'
UBM_covType='diag'
UBM_nComponents = 512
# ------------------------------------------------------------------------------
TDDNN_SAVE_MODELS_DIR = "../BaseDNNTrained/"
TDDNN_SAVE_MODELS_FILE_BASE = "model_tddnn"
TDDNN_TRAIN_PATH = '../BaseDadosNormalizada/TDDNN/'
TDDNN_NUM_WIN_SIZE = 200 
TDDNN_TRAIN_TEST_RATIO = 15
TDDNN_N_EPOCHS = 750
TDDNN_LR = 1e-1 # Initial learning rate
TDDNN_WD = 1e-4 # Weight decay (L2 penalty)
TDDNN_OPT_TYPE = 'sgd' # ex) sgd, adam, adagrad
TDDNN_BATCH_SIZE = 32 # Batch size for training # original 64
TDDNN_VALID_BATCH = 8 # Batch size for validation
TDDNN_USE_SHUFFLE = True
SAVE_MODELS_DIR = 'model_saved'

# ------------------------------------------------------------------------------
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
