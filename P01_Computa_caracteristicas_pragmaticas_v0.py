#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:59:30 2022

@author: adelino
"""
import config as c
import os
from imports.files_utils import list_contend
from imports.acoustic_features import AcousticsFeatures
from imports.AudioData import AudioData
import pickle
from pathlib import Path

       
CONVERT_AUDIO_TO_PROCESS = True
COMPUTE_TRAIN_FEATURES = True

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
    file_list = list_contend(c.AUDIO_TRAIN_PATH,pattern)
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
        
if (COMPUTE_TRAIN_FEATURES):
    audio_file_list = list_contend(c.AUDIO_TRAIN_PATH,(c.FILE_TYPE))
    feture_file_list = list_contend(c.FEATURES_TRAIN_PATH,('.p'))
    file_list.sort()
    print('Inicio do calculo de características:')
    print('Arquivos de treinamento: {}'.format(len(file_list)))
    for idx, audio_file_name in enumerate(audio_file_list):
        ExistFeature, feature_file_path = check_exist_feature_file(audio_file_name,\
                    c.AUDIO_TRAIN_PATH,feture_file_list,c.FEATURES_TRAIN_PATH)
        if (ExistFeature):
            with open(feature_file_path, 'rb') as f:
                features = pickle.load(f)
            print('\tCarregado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
            features.compute_features()
            features.save_preps(c.AUDIO_TRAIN_PATH,c.FEATURES_TRAIN_PATH)
            features.check_nan()
            with open(features.get_feature_file(), 'wb') as f:
                pickle.dump(features,f)
            print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        else:
            if (AudioData(audio_file_name).check(c.FILE_TYPE,c.CODEC,c.CHANNELS,c.SAMPLE_RATE)):
                print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)-1));
                features = AcousticsFeatures(file_name=audio_file_name, )
                features.compute_features()
                features.save_preps(c.AUDIO_TRAIN_PATH,c.FEATURES_TRAIN_PATH)
                features.check_nan()
                with open(features.get_feature_file(), 'wb') as f:
                    pickle.dump(features,f)
                print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        
