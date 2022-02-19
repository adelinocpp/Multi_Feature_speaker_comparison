#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:59:30 2022

@author: adelino
"""
import config as c
from imports.files_utils import list_contend
# from pathlib import Path
from imports.acoustic_features import AcousticsFeatures
from imports.AudioData import AudioData
# import subprocess
import pickle
# import os
        
CONVERT_AUDIO_TO_PROCESS = False
COMPUTE_TRAIN_FEATURES = True

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
            if (AudioData(audio_file_name).check('WAV','PCM','1','8000')):

                    continue
            
            else:
                print('Need test if files are ok')
                file_list[index] = AudioData(audio_file_name).suit('WAV','1','8000')
        except: 
            continue
        
if (COMPUTE_TRAIN_FEATURES):
    file_list = list_contend(c.AUDIO_TRAIN_PATH,('.wav'))
    file_list.sort()
    print('Inicio do calculo de características:')
    print('Arquivos de treinamento: {}'.format(len(file_list)))
    for idx, audio_file_name in enumerate(file_list):
        
        if (AudioData(audio_file_name).check('WAV','PCM','1','8000')):
            print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)-1));
            features = AcousticsFeatures(file_name=audio_file_name, )
            features.compute_features()
            features.save_preps(c.AUDIO_TRAIN_PATH,c.FEATURES_TRAIN_PATH)
            with open(features.get_feature_file(), 'wb') as f:
                pickle.dump(features,f)
            print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        
