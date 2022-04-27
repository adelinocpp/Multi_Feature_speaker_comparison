#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:02:25 2022

@author: adelino
"""

import config as c
import sys
from imports.files_utils import list_contend
import numpy as np
import pickle
import os
from imports.acoustic_features import AcousticsFeatures

# -----------------------------------------------------------------------------
class DataBaseTimeInfo:
    def __init__(self, init_speaker_list=[], init_speaker_time=[]):
        self.speaker_list = init_speaker_list
        self.speaker_time = init_speaker_time       
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
MAKE_SET_CALIBRATION        = True
MAKE_SET_VALIDATION         = True
MAKE_SET_UBM_MODEL          = True
MAKE_SET_LDA_MODEL          = False
MAKE_SET_PLDA_MODEL         = False
MAKE_SET_X_VECTOR_TDDNN     = False
MAKE_SET_X_VECTOR_RESNET    = False
NUM_SETS                    = 7
# -----------------------------------------------------------------------------
pattern = ('.p',)
file_list = list_contend(c.FEATURE_FILE_OUTPUT,pattern)
files_by_set = int(np.ceil(len(file_list)/NUM_SETS))
if not os.path.exists(c.DB_TIME_INFO_FILE):
    dbTimeInfo = DataBaseTimeInfo()  
    
    speaker_list = []
    speaker_time = []
    nFiles = len(file_list)
    for idx, file_feature in enumerate(file_list):
        print("File {:} de {:}".format(idx,nFiles))
        speaker_id = file_feature.split('/')[-2]
        with open(file_feature, 'rb') as f:
            features = pickle.load(f)
        timeF = features.get_feature_time("mfcc")
        
        if (speaker_id in speaker_list):
            idx = speaker_list.index(speaker_id)
            speaker_time[idx] += timeF
        else:
            speaker_list.append(speaker_id)
            speaker_time.append(timeF)
            
    dbTimeInfo.speaker_list = speaker_list
    dbTimeInfo.speaker_time = speaker_time
    ofile = open(c.DB_TIME_INFO_FILE, "wb")
    dill.dump(dbTimeInfo, ofile)
    ofile.close()
    # with open(c.DB_TIME_INFO_FILE, 'wb') as f:
    #     pickle.dump(dbTimeInfo,f)
else:
    ofile = open(c.DB_TIME_INFO_FILE, "rb")
    dbTimeInfo = dill.load(ofile)
    ofile.close()
    
    # with open(c.DB_TIME_INFO_FILE, 'rb') as f:
    #     dbTimeInfo = pickle.load(f)
    print("Arquivo \"{:}\" carregado.".format(c.DB_TIME_INFO_FILE))
    
    
    
    
    