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
# --- FORMANTES
# com base nas caracter

NUM_PLP_FILTERS = 17
NUM_MFCC_FILTERS = 13
NUM_PNCC_FILTERS = 35
NUM_CEPS_COEFS = 13
# PLP_DITHER = False
PLP_SUM_POWER = True