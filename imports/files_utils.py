#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:01:47 2022

@author: adelino
"""
import os
import time
import shutil
from pathlib import Path
# -----------------------------------------------------------------------------
def is_program_installed(program_name):
    check_res = shutil.which(program_name)
    return (check_res is not None)
# -----------------------------------------------------------------------------
def get_time_tag():
    a1 = time.localtime()
    return '{:04}{:02}{:02}{:02}{:02}{:02}'.format(a1.tm_year,a1.tm_mon,a1.tm_mday,a1.tm_hour,a1.tm_min,a1.tm_sec)
# -----------------------------------------------------------------------------
def build_folders_to_save(file_name):
    # if (os.path.isfile(file_name)):
    split_path = file_name.split('/')
    if (split_path[0] == ''):
        split_path = split_path[1:]
    # else:
    #     split_path = file_name.split('/')[1:]
    for idx in range(1,len(split_path)):
        curDir = '/'.join(split_path[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
# ------------------------------------------------------------------------------
def list_contend(folder='./', pattern=()):
    if (len(pattern) > 0):
        pattern = tuple([x.upper() for x in pattern])
        list_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(folder)
                     for name in files
                         if name.upper().endswith(pattern)]
    else:
        list_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(folder)
                     for name in files]
    list_files.sort()
    return list_files
# ------------------------------------------------------------------------------
def check_exist_feature_file(audio_file_name,audio_path,feture_file_list,feature_path):
    file_stem = Path(audio_file_name).stem
    audio_file = Path(audio_file_name).name
    file_feature = audio_file_name.replace(audio_path,feature_path)
    file_feature = file_feature.replace(audio_file,file_stem + '.p')
    result = (file_feature in feture_file_list) and os.path.exists(file_feature)
    if (result == False):
        file_feature = ''
    return result, file_feature

# ------------------------------------------------------------------------------
