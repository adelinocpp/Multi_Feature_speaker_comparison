#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:25:19 2023

@author: adelino
"""
import config as c
import os
from imports.files_utils import list_contend #, check_exist_feature_file, is_program_installed
from multiprocessing import Pool, Lock
import dill
from imports.acoustic_features import AcousticsFeatures

lock = Lock()
MULTI_CORES = True
# =============================================================================
def function_verify_features(feature_file_name, idx, numFiles):
    # ExistFeature, feature_file_path = check_exist_feature_file(audio_file_name,\
    #             c.AUDIO_FILE_INPUT,feature_file_list,c.FEATURE_FILE_OUTPUT)
    # --- gambiarra para a por a lista de caracteristicas no multithreading
    if (os.path.exists(feature_file_name)):
        ofile = open(feature_file_name, "rb")
        features = dill.load(ofile)
        ofile.close()
        has_NAN = features.check_nan()
        lock.acquire()
        if (has_NAN):
            print("Valor NAN encontrado no arquivo: {:}".format(feature_file_name))
        print('Verificado arquivo {:4} de {:4}'.format(idx, numFiles))
        print('{:}'.format(features.get_feature_report()))
        lock.release()
    else:
        lock.acquire()
        print("Arquivo nao encontrado: {:}".format(feature_file_name))
        lock.release()
    return;
# =============================================================================

feature_file_list = list_contend(c.FEATURE_FILE_OUTPUT,(c.RAW_FEAT_EXT,))
# --------------------------------------------------
pool = Pool(os.cpu_count())

numFiles = len(feature_file_list)-1
print("Total de {:} arquivos".format(numFiles+1))
print('Inicio da verificaçao de caracteristicas:')
print('Arquivos de de caracteristicas encontrados: {}'.format(len(feature_file_list)))
if (MULTI_CORES):
    print('MULTITHREADING:')
    input_thread = [[filename, idx,numFiles] for idx, filename in enumerate(feature_file_list)]
    pool.starmap(function_verify_features,input_thread)
else:
    print('SINGLE THREAD:')
    for idx, feature_file_name in enumerate(feature_file_list):
        function_verify_features(feature_file_name,idx,numFiles)
print('Fim do calculo de características:')     