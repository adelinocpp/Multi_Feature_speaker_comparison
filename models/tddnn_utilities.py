#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:52:11 2022

@author: adelino
"""
import dill
from imports.files_utils import list_contend
import sys
import re
import numpy as np
import config as c
import pandas as pd
# ----------------------------------------------------------------------------            
def get_min_loss_model(log_dir):
    start = 0 # Start epoch
    save_model_file_list = list_contend(log_dir,('.pth',))
    save_model_file_list.sort()
    maxLoss = sys.float_info.max
    if (len(save_model_file_list) > 0):
        start = 0
        for file in save_model_file_list:
            values = re.split('_|/|.pth',file)
            idx = int(values[-3])
            loss = float(values[-2])
            if (loss < maxLoss):
                maxLoss = loss
                start = idx
    return start, maxLoss
# ----------------------------------------------------------------------------            
def read_feats_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = list_contend(directory,('.p',)) #find_feats(directory) # filename
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2]) # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-2][4:]) # dataset folder name
    DB['speaker_count'] = DB['speaker_id']  
    # DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2]) # speaker folder name
    # DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # dataset folder name
    # DB['speaker_count'] = DB['speaker_id']
    numEmbeddings = {}
    for idx, element in enumerate(DB['speaker_id']):
        if element in numEmbeddings:
            numEmbeddings[element] += 1
            DB['speaker_count'][idx] = numEmbeddings[element]
        else:
            numEmbeddings[element] = 1
            DB['speaker_count'][idx] = numEmbeddings[element]

    # DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    # logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    # logging.info(DB.head(10))
    return DB, num_speakers
# ----------------------------------------------------------------------------  
def read_MFB(filename, num_win_size=c.TDDNN_NUM_WIN_SIZE):
    ofile = open(filename, "rb")
    featureObj = dill.load(ofile)
    ofile.close()
    feature = featureObj.features["mfcc"].data.T.astype(np.float32)
    label = filename.split('/')[-2]
    if (feature.shape[0] <  c.TDDNN_NUM_WIN_SIZE):
        print("read_MFB error: num_frames < win_size")
        print("filename: {}".format(filename))
        print("feature shape: {}, {}".format(feature.shape[0],feature.shape[1]))

    return feature, label
# ----------------------------------------------------------------------------  
