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
import os
from imports.acoustic_features import AcousticsFeatures
import dill

# -----------------------------------------------------------------------------
class PoolSpeaker:
    def __init__(self, poll_name):
        self.pool_name = poll_name
        self.pool_speakers = np.array([])
        self.pool_time = np.array([])
        self.pool_prop = np.array([])
        
# -----------------------------------------------------------------------------
class SetSpeakesInfo:
    def __init__(self, set_name):
        self.set_id = set_name
        self.percentile = []
        self.bins_all=[]
        self.binsLDA=[]
        self.nLDA=[]
        self.speaker_time = []
        self.speaker_list = []
# -----------------------------------------------------------------------------
class DataBaseTimeInfo:
    def __init__(self, init_speaker_list=np.array([]), init_speaker_time=np.array([]),\
                 init_speaker_sets = np.array([])):
        self.speaker_list = init_speaker_list
        self.speaker_time = init_speaker_time 
        self.speaker_sets = init_speaker_sets
        
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
MAKE_SET_CALIBRATION        = True
MAKE_SET_VALIDATION         = True
MAKE_SET_UBM_MODEL          = True
MAKE_SET_LDA_MODEL          = False
# MAKE_SET_PLDA_MODEL         = False
MAKE_SET_X_VECTOR_TDDNN     = False
MAKE_SET_X_VECTOR_RESNET    = False
SETS = ["LDA","UBM","TDDNN","RESNET","CALIBRATION","VALIDATION"]
NUM_SETS                    = len(SETS)
# -----------------------------------------------------------------------------
NEW_RUN_ALL = True
# -----------------------------------------------------------------------------
pattern = ('.p',)
file_list = list_contend(c.FEATURE_FILE_OUTPUT,pattern)
files_by_set = int(np.ceil(len(file_list)/NUM_SETS))
if (not os.path.exists(c.DB_TIME_INFO_FILE)) or NEW_RUN_ALL:
    dbTimeInfo = DataBaseTimeInfo()  
    speaker_list = np.array([])
    speaker_time = np.array([])
    speaker_db = np.array([])
    nFiles = len(file_list)
    for idx, file_feature in enumerate(file_list):
        print("File {:} de {:}".format(idx,nFiles))
        speaker_id = file_feature.split('/')[-2]
        
        ofile = open(file_feature, "rb")
        features = dill.load(ofile)
        ofile.close()
        # with open(file_feature, 'rb') as f:
        #     features = pickle.load(f)
        timeF = features.get_feature_time("mfcc")
        if (speaker_id in speaker_list):
            idx = np.where(speaker_list==speaker_id)[0][0]
            speaker_time[idx] += timeF
        else:
            speaker_list = np.append(speaker_list,speaker_id)
            speaker_time = np.append(speaker_time,timeF)
            speaker_db = np.append(speaker_db,speaker_id[:4])
    
    
    selTimeIdx = (speaker_time > c.DB_POOL_MIN_TIME).nonzero()[0]
    
    dbTimeInfo.speaker_list = speaker_list[selTimeIdx]
    dbTimeInfo.speaker_time = speaker_time[selTimeIdx]
    dbTimeInfo.speaker_sets = speaker_db[selTimeIdx]
    ofile = open(c.DB_TIME_INFO_FILE, "wb")
    dill.dump(dbTimeInfo, ofile)
    ofile.close()
else:
    ofile = open(c.DB_TIME_INFO_FILE, "rb")
    dbTimeInfo = dill.load(ofile)
    ofile.close()
    print("Arquivo \"{:}\" carregado.".format(c.DB_TIME_INFO_FILE))

if (not os.path.exists(c.DB_TIME_SEPR_FILE)) or NEW_RUN_ALL:
    selectIndex = (np.array(dbTimeInfo.speaker_time) >= 10).nonzero()[0]
    numFilesBySet = len(selectIndex)/NUM_SETS
    segFiles = int(np.floor(numFilesBySet))
    tempSpeakerList = np.array(dbTimeInfo.speaker_list)
    tempSpeakerTime = np.array(dbTimeInfo.speaker_time)
    
    listPool = np.array([])
    for setname in SETS:
        pool = PoolSpeaker(setname)
        if (setname == 'LDA'):
            LDAlist = np.array([])
            LDAtime = np.array([])
            pool.pool_prop = [int(np.floor(0.5*numFilesBySet)),\
                      int(np.floor(0.3*numFilesBySet)),\
                      int(np.floor(0.2*numFilesBySet))]
                
            for idx, nF in enumerate(pool.pool_prop):
                idxFiles = (tempSpeakerTime >= 30*(idx*2+3)).nonzero()[0]
                selFiles = idxFiles[np.random.permutation(len(idxFiles))[:nF]]
                LDAlist = np.append(LDAlist,tempSpeakerList[selFiles])
                LDAtime  = np.append(LDAtime,tempSpeakerTime[selFiles])
                tempSpeakerTime = np.delete(tempSpeakerTime,selFiles)
                tempSpeakerList = np.delete(tempSpeakerList,selFiles)
            pool.pool_speakers = LDAlist
            pool.pool_time = LDAtime
        elif (setname == 'UBM'):
            idxFiles = (tempSpeakerTime >= 60).nonzero()[0]
            selFiles = idxFiles[np.random.permutation(len(idxFiles))[:segFiles]]
            pool.pool_speakers = tempSpeakerList[selFiles]
            pool.pool_time  = tempSpeakerTime[selFiles]
            tempSpeakerTime = np.delete(tempSpeakerTime,selFiles)
            tempSpeakerList = np.delete(tempSpeakerList,selFiles)
        else:
            idxFiles = (tempSpeakerTime >= 10).nonzero()[0]
            selFiles = idxFiles[np.random.permutation(len(idxFiles))[:segFiles]]
            pool.pool_speakers = tempSpeakerList[selFiles]
            pool.pool_time  = tempSpeakerTime[selFiles]
            tempSpeakerTime = np.delete(tempSpeakerTime,selFiles)
            tempSpeakerList = np.delete(tempSpeakerList,selFiles)
        
        listPool = np.append(listPool,pool)
        
    ofile = open(c.DB_TIME_SEPR_FILE, "wb")
    dill.dump(listPool, ofile)
    ofile.close()
else:
    ofile = open(c.DB_TIME_SEPR_FILE, "rb")
    listPool = dill.load(ofile)
    ofile.close()
    print("Arquivo \"{:}\" carregado.".format(c.DB_TIME_SEPR_FILE))
