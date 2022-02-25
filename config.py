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
# Etapa 1 calculo de caracteristicas pragmaticas
# Audios de entrada
AUDIO_TRAIN_PATH = '../BaseDadosAudios/Treinamento/'
AUDIO_TESTI_PATH = '../BaseDadosAudios/Teste/'
AUDIO_VALID_PATH = '../BaseDadosAudios/Validacao/'
# Caracteristicas calculadas
FEATURES_TRAIN_PATH = '../BaseDadosFeatures/Treinamento/'
FEATURES_TESTI_PATH = '../BaseDadosFeatures/Teste/'
FEATURES_VALID_PATH = '../BaseDadosFeatures/Validacao/'
# ------------------------------------------------------------------------------
WIN_SIZE = 0.025
STEP_SIZE = 0.01
# ------------------------------------------------------------------------------
# ==============================================================================
# Dados para calculo das características
# --- PRE-ENFASE
PRE_ENPHASIS_COEF = 0.975
# --- FORMANTES
# com base nas caracteriticas do audio e SAMPLE_RATE
NUM_CEPS_COEFS = 13
# --- MFCC
MFCC_NUM_FILT = 13
# --- PLP and RASPA-PLP
PLP_NUM_FILTERS = 17
PLP_SUM_POWER = True
PLP_LIFTER_COEF = 0.6
# --- PNCC
PNCC_NUM_FILTERS = 35
# PLP_DITHER = False
