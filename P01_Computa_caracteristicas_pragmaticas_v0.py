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
# import audio_metadata
# import subprocess
import pickle
# import os
from pathlib import Path
import subprocess

class AudioData:
    def __init__(self, audioFullPath):
        self.suffix = audioFullPath.split('/')[-1].split('.')[-1].upper()
        cmdString = 'mediainfo --Inform=\"Audio;%Format%\" ' + audioFullPath
        returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        self.codec =  returnStr.read().decode()[0:-1]
        cmdString = 'mediainfo --Inform=\"Audio;%Channels%\" ' + audioFullPath
        returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        self.channels =  returnStr.read().decode()[0:-1]
        cmdString = 'mediainfo --Inform=\"Audio;%SamplingRate%\" ' + audioFullPath
        returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        self.sr =  returnStr.read().decode()[0:-1]
    def check(self,ext,codec,channels,sr):
        return (self.sr == sr) and (self.codec == codec) and \
            (self.channels == channels) and (self.suffix == ext)


CONVERT_AUDIO_TO_PROCESS = True
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
                    print('File looks ok')
                    continue
            # if (inputAudioData.sr == '8000') and (inputAudioData.codec == 'PCM') and\
            #     (inputAudioData.channels == '1') and (strExt == 'WAV'):
            else:
                print('Need test if files are ok')
                dir_path = Path(audio_file_name).parent
                basename = Path(audio_file_name).stem
                ext = Path(audio_file_name).suffix
                # --- rename file for temp.wav
                temp_file_name = dir_path.as_posix() + '/temp' + ext
                cmdString = 'mv {:} {:}'.format(audio_file_name,temp_file_name)
                returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
                new_file_name = dir_path.as_posix() + '/' +  basename + '.wav'
                # --- Convert file
                cmdString = 'sox {:} -c 1 -r 8000 -e signed-integer -b 16 {:}'.format(temp_file_name,new_file_name)
                returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
                file_list[index] = new_file_name
                # --- remove temp file
                cmdString = 'rm {:}'.format(temp_file_name)
                returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
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
        