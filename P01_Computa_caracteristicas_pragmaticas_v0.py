#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:59:30 2022

@author: adelino
"""
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import config as c
import os
from imports.files_utils import list_contend
from imports.acoustic_features import AcousticsFeatures
from imports.AudioData import AudioData
import pickle
from pathlib import Path
import multiprocessing as mp
import time
import dill

CONVERT_AUDIO_TO_PROCESS = True
COMPUTE_FEATURES = True

# -----------------------------------------------------------------------------
def check_exist_feature_file(audio_file_name,audio_path,feture_file_list,feature_path):
    file_stem = Path(audio_file_name).stem
    audio_file = Path(audio_file_name).name
    file_feature = audio_file_name.replace(audio_path,feature_path)
    file_feature = file_feature.replace(audio_file,file_stem + '.p')
    result = (file_feature in feture_file_list) and os.path.exists(file_feature)
    if (result == False):
        file_feature = ''
    return result, file_feature

# -----------------------------------------------------------------------------
def thread_compute_features(obj_features, osNiceVal):
    # pid = os.getpid()
    # os.system("sudo renice -n -5 -p " + str(pid))
    
    obj_features.compute_features()
    obj_features.save_preps(c.AUDIO_FILE_INPUT,c.FEATURE_FILE_OUTPUT)
    obj_features.check_nan()
    ofile = open(obj_features.get_feature_file(), "wb")
    dill.dump(obj_features, ofile)
    ofile.close()

    # with open(obj_features.get_feature_file(), 'wb') as f:
    #     pickle.dump(obj_features,f)
    print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))        
# -----------------------------------------------------------------------------

def check_exist_feature_file(audio_file_name,audio_path,feture_file_list,feature_path):
    file_stem = Path(audio_file_name).stem
    audio_file = Path(audio_file_name).name
    file_feature = audio_file_name.replace(audio_path,feature_path)
    file_feature = file_feature.replace(audio_file,file_stem + '.p')
    result = (file_feature in feture_file_list) and os.path.exists(file_feature)
    if (result == False):
        file_feature = ''
    return result, file_feature

if (CONVERT_AUDIO_TO_PROCESS):
    pattern = ('.3gp','.aa','.aac','.aax','.act','.aiff','.amr','.ape','.au',
               '.awb','.dct','.dss','.dvf','.flac','.gsm','.iklax','.ivs',
               '.m4a','.m4b','.m4p','.mmf','.mp3','.mpc','.msv','.nmf','.nsf',
               '.ogg','.oga','.mogg','.opus','.ra','.rm','.raw','.sln','.tta',
               '.vox','.wav','.wma','.wv','.webm','.8svx')
    file_list = list_contend(c.AUDIO_FILE_INPUT,pattern)
    for index, audio_file_name in enumerate(file_list):
        try:
            strExt = audio_file_name.split('/')[-1].split('.')[-1].upper()
            inputAudioData = AudioData(audio_file_name)
            if (AudioData(audio_file_name).check(c.FILE_TYPE,c.CODEC,c.CHANNELS,c.SAMPLE_RATE)):
                    continue
            else:
                print('Need test if files are ok')
                file_list[index] = AudioData(audio_file_name).suit(c.FILE_TYPE,c.CHANNELS,c.SAMPLE_RATE)
        except:
            continue



if (COMPUTE_FEATURES):
    audio_file_list = list_contend(c.AUDIO_FILE_INPUT,(c.FILE_TYPE))
    feature_file_list = list_contend(c.FEATURE_FILE_OUTPUT,('.p'))
    
    num_cores = mp.cpu_count()        
    set_cores = list([])
    for imp in range(0,num_cores):
        set_cores.append(mp.Process())    
    num_free_cores =  num_cores - sum([core.is_alive() for core in set_cores])
    print('Inicio do calculo de características:')
    print('Arquivos para o calculo de caracterisitcas: {}'.format(len(file_list)))
    for idx, audio_file_name in enumerate(audio_file_list):
        
        
        ExistFeature, feature_file_path = check_exist_feature_file(audio_file_name,\
                    c.AUDIO_FILE_INPUT,feature_file_list,c.FEATURE_FILE_OUTPUT)
        # Verifica se arquivo já existe para atualizar os calculos
        if (ExistFeature):
            with open(feature_file_path, 'rb') as f:
                features = pickle.load(f)
            print('\tCarregado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
            # features.compute_features()
            # features.save_preps(c.AUDIO_FILE_INPUT,c.FEATURE_FILE_OUTPUT)
            # features.check_nan()
            # with open(features.get_feature_file(), 'wb') as f:
            #     pickle.dump(features,f)
            # print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        else:
            if (AudioData(audio_file_name).check(c.FILE_TYPE,c.CODEC,c.CHANNELS,c.SAMPLE_RATE)):
                print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)-1))
                features = AcousticsFeatures(file_name=audio_file_name, )
                # features.compute_features()
                # features.save_preps(c.AUDIO_FILE_INPUT,c.FEATURE_FILE_OUTPUT)
                # features.check_nan()
                # with open(features.get_feature_file(), 'wb') as f:
                #     pickle.dump(features,f)
                # print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        
        num_free_cores =  num_cores - sum([core.is_alive() for core in set_cores])
        if (num_free_cores == 0):
            print("Waiting...")
            while  (num_free_cores == 0):
                time.sleep(2)
                num_free_cores =  num_cores - sum([core.is_alive() for core in set_cores])
        else:
            for idxMP, objMP in enumerate(set_cores):
                if (not objMP.is_alive()):
                    print('Multiprocessing core {:}, arquivo {:} free {:}'.format(idxMP,idx,num_free_cores))
                    newMP = mp.Process(target=thread_compute_features, args=(features,os.nice(0),))
                    newMP.start()
                    set_cores[idxMP] = newMP
                    num_free_cores =  num_cores - sum([core.is_alive() for core in set_cores])
                    break
        
          