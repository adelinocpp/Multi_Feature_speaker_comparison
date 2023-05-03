#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:02:25 2022

@author: adelino
"""

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
        self.pool_speakers = []
        self.pool_time = []
        self.pool_prop = []
        
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
    def __init__(self, init_speaker_list=[], init_speaker_time=[],\
                 init_speaker_sets = []):
        self.speaker_list = init_speaker_list
        self.speaker_time = init_speaker_time 
        self.speaker_sets = init_speaker_sets
        
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
MAKE_SET_CALIBRATION        = True
MAKE_SET_VALIDATION         = True
MAKE_SET_UBM_MODEL          = True
MAKE_SET_LDA_MODEL          = True
# MAKE_SET_PLDA_MODEL         = False
MAKE_SET_X_VECTOR_TDDNN     = True
MAKE_SET_X_VECTOR_RESNET    = True
NUM_SETS                    = len(c.SPEAKER_SETS)
# -----------------------------------------------------------------------------
NEW_RUN_ALL         = False
COMPUTE_PERCENTIL   = False
# -----------------------------------------------------------------------------
pattern = (c.RAW_FEAT_EXT,)
file_list = list_contend(c.FEATURE_FILE_OUTPUT,pattern)
ofile = open(c.FEATURE_FILE_LIST, "wb")
dill.dump(file_list, ofile)
ofile.close()


files_by_set = int(np.ceil(len(file_list)/NUM_SETS))
if (not os.path.exists(c.DB_TIME_INFO_FILE)) or NEW_RUN_ALL:
    print("Inicio do calculo do arquivo de tempo...")
    dbTimeInfo = DataBaseTimeInfo()  
    speaker_list = []
    speaker_time = []
    speaker_db = []
    nFiles = len(file_list)
    for idx, file_feature in enumerate(file_list):
        #if ((idx % 100) == 0):
        print("File {:>5} de {:>5}".format(idx,nFiles))
        speaker_id = file_feature.split('/')[-2]
        ofile = open(file_feature, "rb")
        features = dill.load(ofile)
        ofile.close()
        try:
            timeF = features.get_feature_time("mfcc")
            if (speaker_id in speaker_list):
                idx = speaker_list.index(speaker_id)
                speaker_time[idx] += timeF
            else:
                speaker_list.append(speaker_id)
                speaker_time.append(timeF)
                speaker_db.append(speaker_id[:4])
        except: 
            print('Problema no arquivo {:}, id {:}'.format(file_feature.split("/")[-1],speaker_id))
    dbTimeInfo.speaker_list = speaker_list
    dbTimeInfo.speaker_time = speaker_time
    dbTimeInfo.speaker_sets = speaker_db
    ofile = open(c.DB_TIME_INFO_FILE, "wb")
    dill.dump(dbTimeInfo, ofile)
    ofile.close()
else:
    print("Arquivo de tempo ja existe...")
    ofile = open(c.DB_TIME_INFO_FILE, "rb")
    dbTimeInfo = dill.load(ofile)
    ofile.close()
    print("Arquivo \"{:}\" carregado.".format(c.DB_TIME_INFO_FILE))


if (COMPUTE_PERCENTIL):
    # esta parte estava comentada. Passar para o bloco de visualizaçao
    listSetSpeaker = np.unique(dbTimeInfo.speaker_sets)
    binsAll = [0,20,40,80,160,200,280,320,400,np.Inf]
    binsLDA = [0,30,60,90,120,240,300,np.Inf]
    
    samplesByLimAll, _ = np.histogram(dbTimeInfo.speaker_time,bins=binsAll)
    samplesByLimLDA3, _ = np.histogram(dbTimeInfo.speaker_time,bins=np.multiply(3,binsLDA))
    samplesByLimLDA5, _ = np.histogram(dbTimeInfo.speaker_time,bins=np.multiply(5,binsLDA))
    samplesByLimLDA7, _ = np.histogram(dbTimeInfo.speaker_time,bins=np.multiply(7,binsLDA))
    listPercentile = []
    for set_speaker in listSetSpeaker:
        idxSet = np.array([idSS == set_speaker for idSS in dbTimeInfo.speaker_sets]).nonzero()[0]
        setFeaturesSpeakes = SetSpeakesInfo(set_speaker)
        setFeaturesSpeakes.speaker_time = np.array(dbTimeInfo.speaker_time)[idxSet]
        setFeaturesSpeakes.speaker_list = np.array(dbTimeInfo.speaker_list)[idxSet]
        setFeaturesSpeakes.bins_all= binsAll
        setFeaturesSpeakes.binsLDA = binsLDA
        setFeaturesSpeakes.nLDA = [3,5,7]
        setFeaturesSpeakes.samplesByLimAll, _ = np.histogram(setFeaturesSpeakes.speaker_time,bins=binsAll)
        setFeaturesSpeakes.samplesByLimLDA3, _ = np.histogram(setFeaturesSpeakes.speaker_time,bins=np.multiply(3,binsLDA))
        setFeaturesSpeakes.samplesByLimLDA5, _ = np.histogram(setFeaturesSpeakes.speaker_time,bins=np.multiply(5,binsLDA))
        setFeaturesSpeakes.samplesByLimLDA7, _ = np.histogram(setFeaturesSpeakes.speaker_time,bins=np.multiply(7,binsLDA))
        setFeaturesSpeakes.percentile = np.percentile(setFeaturesSpeakes.speaker_time,\
                                                    [10,20,30,40,50,60,70,80,90])
        listPercentile.append(setFeaturesSpeakes)
    


if ((not os.path.exists(c.DB_TIME_SEPR_FILE)) or NEW_RUN_ALL):
    print("Montando conjuntos de locutores...")
    selectIndex = (np.array(dbTimeInfo.speaker_time) >= 10).nonzero()[0]
    numFilesBySet = len(selectIndex)/NUM_SETS
    segFiles = int(np.floor(numFilesBySet))
    tempSpeakerList = np.array(dbTimeInfo.speaker_list)
    tempSpeakerTime = np.array(dbTimeInfo.speaker_time)
    
    listPool = np.array([])
    for setname in c.SPEAKER_SETS:
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
    print("Conjuntos de locutores ja existe...")
    ofile = open(c.DB_TIME_SEPR_FILE, "rb")
    listPool = dill.load(ofile)
    ofile.close()
    print("Arquivo \"{:}\" carregado.".format(c.DB_TIME_SEPR_FILE))
