#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Adelino P. Silva
email: adelinocpp@yahoo.com
Ultima modificacao: 22/12/2022
Calculo das caracteristicas dos arquivos de audio
- opcao multithreading
- codigo enxuto
TODO:
    - implemetar o calculo das caracteristicas com opçoes do que calcular
    e.g. apenas RASTA-PLP, MFCC etc...
    - atualizar calculos
"""
import config as c
import os
from imports.files_utils import list_contend, check_exist_feature_file, is_program_installed
from imports.acoustic_features import AcousticsFeatures
from imports.AudioData import AudioData
# import pickle
# from pathlib import Path
# import multiprocessing as mp
import dill
import sys
from multiprocessing import Pool, Lock

# -----------------------------------------------------------------------------
CONVERT_AUDIO_TO_PROCESS = True
COMPUTE_FEATURES = True
MULTI_CORES = True
lock = Lock()
# =============================================================================
def function_compute_features(audio_file_name, idx, numFiles,update=False):
    # ExistFeature, feature_file_path = check_exist_feature_file(audio_file_name,\
    #             c.AUDIO_FILE_INPUT,feature_file_list,c.FEATURE_FILE_OUTPUT)
    # --- gambiarra para a por a lista de caracteristicas no multithreading
    ExistFeature, feature_file_path = check_exist_feature_file(audio_file_name,\
                c.AUDIO_FILE_INPUT,c.CURRENT_RAW_FEATURE_FILE_LIST,c.FEATURE_FILE_OUTPUT)
    # --------------------------------------------------
    
    # Verifica se arquivo já existe para atualizar os calculos
    print('Iniciado arquivo   {:5} de {:5}'.format(idx, numFiles))
    if (ExistFeature and update):
        ofile = open(feature_file_path, "rb")
        features = dill.load(ofile)
        ofile.close()
        print('Carregado arquivo {:5} de {:5}'.format(idx, numFiles))
    else:
        if (AudioData(audio_file_name).check(c.FILE_TYPE,c.CODEC,c.CHANNELS,c.SAMPLE_RATE)):
            features = AcousticsFeatures(file_name=audio_file_name, )
        else:
            print("Problemas com arquivo {:}".format(audio_file_name))
            print("Verifique se:") 
            print("             o o arquivo é do tipo {:}.".format(c.FILE_TYPE))
            print("             o codec é {:}.".format(c.CODEC))
            print("             se o número de canais é {:}.".format(c.CHANNELS))
            print("             a taxa de amostragem é {:}.".format(c.SAMPLE_RATE))
            print("Ou altere a opção CONVERT_AUDIO_TO_PROCESS para True")
            return;
    if (not ExistFeature or update):
        # lock.acquire()
        #  salva arquivo de caracteristicas
        features.compute_features()
        features.save_preps(c.AUDIO_FILE_INPUT,c.FEATURE_FILE_OUTPUT)
        features.check_nan()
        ofile = open(features.get_feature_file(), "wb")
        dill.dump(features, ofile)
        ofile.close()
        # feature_file_list = c.CURRENT_RAW_FEATURE_FILE_LIST
        # feature_file_list.append(ofile)
        # c.CURRENT_RAW_FEATURE_FILE_LIST = feature_file_list
        # lock.release()
    print('Finalizado arquivo {:5} de {:5}'.format(idx, numFiles))
# =============================================================================
def function_convert_file(audio_file_name, index, numFiles):
    try:
        # strExt = audio_file_name.split('/')[-1].split('.')[-1].upper()
        # inputAudioData = AudioData(audio_file_name)
        if (AudioData(audio_file_name).check(c.FILE_TYPE,c.CODEC,c.CHANNELS,c.SAMPLE_RATE)):
            return audio_file_name
        else:
            print('Need test if files are ok')
            return AudioData(audio_file_name).suit(c.FILE_TYPE,c.CHANNELS,c.SAMPLE_RATE)
    except:
        print('Erro na conversao do arquivo {:}'.format(audio_file_name))
        return audio_file_name
        
    print("Fim arquivo {:5} de {:5}.".format(index,numFiles))    
# =============================================================================
pool = Pool(os.cpu_count())
if (CONVERT_AUDIO_TO_PROCESS):
    if (is_program_installed("mediainfo")):
        print("Processo de conversão iniciado...")
        file_list = list_contend(c.AUDIO_FILE_INPUT,c.PATTERN_AUDIO_EXT)
        numFiles = len(file_list)-1
        print("Total de {:} arquivos".format(numFiles+1))
        if (MULTI_CORES):
            input_thread = [[audio, idx,numFiles] for idx, audio in enumerate(file_list)]
            file_list = pool.starmap(function_convert_file,input_thread)
        else:
            for index, audio_file_name  in enumerate(file_list):
                file_list[index] = function_convert_file(audio_file_name,index,numFiles)
        print("Processo de conversão finalizado.")
    else:
        print("Processo de conversão interrompido.")
        print("mediainfo não está instalado.")
        print("Para continuar instale mediainfo com o comando :")
        print("\tsudo apt-get install mediainfo")
        COMPUTE_FEATURES = False
    
if (COMPUTE_FEATURES):
    audio_file_list = list_contend(c.AUDIO_FILE_INPUT,(c.FILE_TYPE))
    feature_file_list = list_contend(c.FEATURE_FILE_OUTPUT,(c.RAW_FEAT_EXT,))
    # --- gambiarra para a por a lista de caracteristicas no multithreading
    c.CURRENT_RAW_FEATURE_FILE_LIST = feature_file_list
    # --------------------------------------------------
    numFiles = len(audio_file_list)-1
    print("Total de {:} arquivos".format(numFiles+1))
    print('Inicio do calculo de caracteristicas:')
    print('Arquivos de áudios para o calculo de caracteristicas: {:}'.format(len(audio_file_list)))
    print('Arquivos de de caracteristicas encontrados: {:}'.format(len(feature_file_list)))
    if (MULTI_CORES):
        print('MULTITHREADING:')
        input_thread = [[audio, idx,numFiles] for idx, audio in enumerate(audio_file_list)]
        pool.starmap(function_compute_features,input_thread)
    else:
        print('SINGLE THREAD:')
        for idx, audio_file_name in enumerate(audio_file_list):
            function_compute_features(audio_file_name,idx,numFiles)
    print('Fim do calculo de características:')     
    # feature_file_list = c.CURRENT_RAW_FEATURE_FILE_LIST
    # feature_file_list = feature_file_list.unique()
    
# =============================================================================
