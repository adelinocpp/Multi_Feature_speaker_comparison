#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:56:06 2022

@author: adelino
"""
import config as c
from imports.files_utils import list_contend
import dill
import numpy as np
from imports.SpheringSVD import SpheringSVD as Sphering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 


LDA_iVECTOR_RUN = True
LDA_xVECTOR_TDDNN_RUN = True
LDA_xVECTOR_RESNET_RUN = True

pattern = (".pth",)

if (LDA_iVECTOR_RUN):
    lda_ivectos_file_list = list_contend(c.IVECTOR_MODEL_LDA_DIR, pattern)
    nFiles = len(lda_ivectos_file_list)
    y = np.zeros((nFiles,))
    
    for idx, file_name in enumerate(lda_ivectos_file_list):
        speaker_id = file_name.split('/')[-2]
        ofile = open(file_name, "rb")
        spk_vector = dill.load(ofile)
        ofile.close()
        y[idx] = int(speaker_id);
        if (idx == 0):
            embedding_size = spk_vector.shape[0]
            X = np.zeros((nFiles,embedding_size))
        X[idx,:] = np.array(spk_vector)
    # --- processo de Sphearing ----------------------------------------------------
    SpheModel = Sphering.SpheringSVD()
    SpheModel.fit(X)
    Xsph = SpheModel.transform(X)
    LDAModel = LinearDiscriminantAnalysis()
    LDAModel.fit(Xsph, y)
    

