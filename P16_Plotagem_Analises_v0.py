#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:17:58 2022

@author: adelino
"""

import config as c
import dill
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.mixture import GaussianMixture
from scipy import stats

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from sklearn.metrics import confusion_matrix

from fastkde.fastKDE import pdf

plt.close('all')
plt.style.use('tableau-colorblind10')

class Conf_Matrix_Section:
    def __init__(self):
        self.vecBAS = []
        self.vecTPR = []
        self.vecFPR = []
        self.vecTNR = []
        self.vecFNR = []
        self.vecACC = []
        self.vecBAC = []
# -----------------------------------------------------------------------------
def confusion_matrix_section(vecEspected,vecResult,vec_time, vec_lim):
    retBas = np.empty([len(vec_lim)+1,], dtype=object)
    cmr = Conf_Matrix_Section()
    for i in range(0,len(vec_lim) +1):
        if (i == 0):
            idx  = (vec_time < vec_lim[i]).nonzero()[0]        
            cmr.vecBAS = np.append(cmr.vecBAS,'t<{:}'.format(vec_lim[i]))
        elif (i == len(vec_lim)):
            idx = (vec_time >= vec_lim[i-1]).nonzero()[0]
            cmr.vecBAS = np.append(cmr.vecBAS,'{:}<t'.format(vec_lim[i-1]))
        else:
            idx  = np.multiply(vec_time >=  vec_lim[i-1],vec_time < vec_lim[i]).nonzero()[0]
            cmr.vecBAS = np.append(cmr.vecBAS,'{:}<t<{:}'.format(vec_lim[i-1],vec_lim[i]))
        vec_expected = vecEspected[idx]
        vec_result = vecResult[idx]
        idxSameSpeaker = (vec_expected == 1).nonzero()[0]
        idxDiffSpeaker = (vec_expected == 0).nonzero()[0]
        p = len(idxSameSpeaker)
        n = len(idxDiffSpeaker)
        
        tn, fp, fn, tp = confusion_matrix(vec_expected,vec_result).ravel()
        cmr.vecTPR = np.append(cmr.vecTPR,100*tp/p)
        cmr.vecTNR = np.append(cmr.vecTNR,100*tn/n)
        cmr.vecFPR = np.append(cmr.vecFPR,100*fp/n)
        cmr.vecFNR = np.append(cmr.vecFNR,100*fn/p)
        cmr.vecACC = np.append(cmr.vecACC,100*(tp+tn)/(n+p))
        cmr.vecBAC = np.append(cmr.vecBAC,50*(tp/p+tn/n))
    return cmr
# -----------------------------------------------------------------------------
def resume_confusion_matrix(vecEspected,vecResult,labelType,labelThr,eer):
    numComp = len( vecEspected)
    idxSameSpeaker = (vecEspected == 1).nonzero()[0]
    idxDiffSpeaker = (vecEspected == 0).nonzero()[0]
    p = len(idxSameSpeaker)
    n = len(idxDiffSpeaker)
    
    tn, fp, fn, tp = confusion_matrix(vecEspected,vecResult).ravel()
    TPR = tp/p
    TNR = tn/n
    FPR = fp/n
    FNR = fn/p
    if (FNR == 0):
        FNR = 1e-5
    if (FPR == 0):
        FPR = 1e-5
    DOF = TPR*TNR/(FPR*FNR)
    print("Resumo por {:} em {:}:".format(labelType,labelThr))
    print("N: {:,}; EER: {:3.2f}%; Prev: {:3.2f}%; DOR: {:3.2f};".format(
                    n+p, 100*eer, 100*p/(n+p), DOF))
    print("TP: {:3.2f}%; TN: {:3.2f}%; FP: {:3.2f}%; FN: {:3.2f}%; AC: {:3.2f}%; BA: {:3.2f}%.".format(
        100*TPR,100*TNR, 100*FPR,100*FNR,100*(tp+tn)/(n+p),50*(TPR+TNR)))    
# -----------------------------------------------------------------------------
def plot_score(vecEspected,vecScore,thr,labelType):
    numComp = len( vecEspected)
    stdScore = np.std(vecScore)
    meanScore = np.mean(vecScore)
    scrMin = meanScore - 3*stdScore # np.min(vecScore) - 0.1*stdScore
    scrMax = np.max(vecScore) # - 0.1*stdScore # np.max(vecScore) + 0.1*stdScore
    scrBase = np.linspace(scrMin,scrMax,5000)
    idxSameSpeaker = (vecEspected == 1).nonzero()[0]
    idxDiffSpeaker = (vecEspected == 0).nonzero()[0]
    ssScore = vecScore[idxSameSpeaker]
    dsScore = vecScore[idxDiffSpeaker]
    
    yssScore, xss = pdf(ssScore)
    ydsScore, xds = pdf(dsScore)
    ymax = np.max(np.concatenate((yssScore,ydsScore)))
    
    fig = plt.figure(figsize =(6, 6))
    T = ["Mesmo locutor", "locutor diferente","limiar decisao"]
    
    plt.title("Distribução de score em {:}".format(labelType))
    plt.ylabel("Densidade")
    plt.xlabel('Score')
    plt.plot(xss,yssScore, linewidth=2)
    plt.plot(xds,ydsScore, linewidth=2)
    plt.plot([thr,thr],[0,1.1*ymax], 'k-.',linewidth=2)
    plt.xlim([scrMin,scrMax])
    plt.ylim([0,1.1*ymax])
    plt.grid(color='g', linestyle='-.', linewidth=0.5)
    plt.legend(T)
# -----------------------------------------------------------------------------
def lista_tempo_audio(vec_time, vec_lim):
    retList = np.empty([len(vec_lim)+1,], dtype=object)
    
    for i in range(0,len(vec_lim) +1):
        if (i == 0):
            idx  = (vec_time < vec_lim[i]).nonzero()[0]        
        elif (i == len(vec_lim)):
            idx = (vec_time >= vec_lim[i-1]).nonzero()[0]
        else:
            idx  = np.multiply(vec_time >=  vec_lim[i-1],vec_time < vec_lim[i]).nonzero()[0]
        retList[i] = idx
    return retList
# -----------------------------------------------------------------------------
Analise_GMM_UBM = True
Analise_ivector = True
Analise_xvector_TDDNN = True
Analise_xvector_RESNET = True
timesLim = np.array([30,60,120,180,240,300,500])

if os.path.exists(c.GMM_RESULT_PARAM) and os.path.exists(c.GMM_VALIDATE_RESULT) and Analise_GMM_UBM:
    ofile = open(c.GMM_RESULT_PARAM, "rb")
    gmm_result_param = dill.load(ofile)
    ofile.close();
    ofile = open(c.GMM_VALIDATE_RESULT, "rb")
    gmm_validate_result = dill.load(ofile)
    ofile.close();
    
    # -------------------------------------------------------------------------
    resume_confusion_matrix(gmm_validate_result['expected'],gmm_validate_result['byLogistic'],
                            "GMM-UBM","Regressão logistica",gmm_result_param['eer'] )
    
    # plot_score(gmm_validate_result['expected'],gmm_validate_result['score'],gmm_result_param['thresh'],"GMM-UBM")
    
    cmsGMM = confusion_matrix_section(gmm_validate_result['expected'],gmm_validate_result['byLogistic'],
                             gmm_validate_result['timeAudio'],timesLim)
    
    
    # listaTempo = lista_tempo_audio(gmm_validate_result['timeAudio'],timesLim)
    # for idx in range(0,len(timesLim)+1):
    #     idxL = listaTempo[idx]
    #     vec_expected = gmm_validate_result['expected'][idxL]
    #     vec_result = gmm_validate_result['byLogistic'][idxL]
    #     vec_score = gmm_validate_result['score'][idxL]
    #     tag_Classificador = "Regressão logistica"
    #     eer_val = gmm_result_param['eer']
    #     thr_val = gmm_result_param['thresh']
    #     if (idx == 0):
    #         label_Result = "GMM-UBM até {:} seg.".format(timesLim[idx])
    #     elif (idx == len(timesLim)):
    #         label_Result = "GMM-UBM acima de {:} seg.".format(timesLim[idx-1])
    #     else:
    #         label_Result = "GMM-UBM entre {:} e {:} seg.".format(timesLim[idx-1],timesLim[idx])
    #     resume_confusion_matrix(vec_expected,vec_result,
    #                             label_Result,tag_Classificador,eer_val)
    
# -----------------------------------------------------------------------------
if os.path.exists(c.IVECTOR_RESULT_PARAM) and os.path.exists(c.IVECTOR_VALIDATE_RESULT) and Analise_ivector:
    ofile = open(c.IVECTOR_RESULT_PARAM, "rb")
    ivector_result_param = dill.load(ofile)
    ofile.close();
    ofile = open(c.IVECTOR_VALIDATE_RESULT, "rb")
    ivector_validate_result = dill.load(ofile)
    ofile.close();

    resume_confusion_matrix(ivector_validate_result['expected'],ivector_validate_result['byLogistic'],
                            "ivector","Regressão logistica",ivector_result_param['eer'])
    # plot_score(ivector_validate_result['expected'],ivector_validate_result['score'],ivector_result_param['thresh'],"ivector")
    
    cmsiVEC = confusion_matrix_section(ivector_validate_result['expected'],ivector_validate_result['byLogistic'],
                             ivector_validate_result['timeAudio'],timesLim)
# -----------------------------------------------------------------------------    
if os.path.exists(c.XVECTOR_TDDNN_RESULT_PARAM) and os.path.exists(c.XVECTOR_TDDNN_VALIDATE_RESULT) and Analise_xvector_TDDNN:
    ofile = open(c.XVECTOR_TDDNN_RESULT_PARAM, "rb")
    tddnn_result_param = dill.load(ofile)
    ofile.close()
    ofile = open(c.XVECTOR_TDDNN_VALIDATE_RESULT, "rb")
    tddnn_validate_result = dill.load(ofile)
    ofile.close();
    
    resume_confusion_matrix(tddnn_validate_result['expected'],tddnn_validate_result['byLogistic'],
                            "TDDNN","Regressão logistica",tddnn_result_param['eer'])
    # plot_score(tddnn_validate_result['expected'],tddnn_validate_result['score'],tddnn_result_param['thresh'],"TDDNN")

    cmsxTDDNN = confusion_matrix_section(tddnn_validate_result['expected'],tddnn_validate_result['byLogistic'],
                             tddnn_validate_result['timeAudio'],timesLim)
# -----------------------------------------------------------------------------    
if os.path.exists(c.XVECTOR_RESNET_RESULT_PARAM) and os.path.exists(c.XVECTOR_RESNET_VALIDATE_RESULT) and Analise_xvector_RESNET:
    ofile = open(c.XVECTOR_RESNET_RESULT_PARAM, "rb")
    resnet_result_param = dill.load(ofile)
    ofile.close()
    ofile = open(c.XVECTOR_RESNET_VALIDATE_RESULT, "rb")
    resnet_validate_result = dill.load(ofile)
    ofile.close();
    
    resume_confusion_matrix(resnet_validate_result['expected'],resnet_validate_result['byLogistic'],
                            "RESNET","Regressão logistica",resnet_result_param['eer'])
    # plot_score(resnet_validate_result['expected'],resnet_validate_result['score'],resnet_result_param['thresh'],"RESNET")

    cmsxRESNET = confusion_matrix_section(resnet_validate_result['expected'],resnet_validate_result['byLogistic'],
                             resnet_validate_result['timeAudio'],timesLim)
# -----------------------------------------------------------------------------    
T = []
fig = plt.figure(figsize =(9, 6))
plt.title("Taxas de verdadeiro possitivo")
plt.ylabel("pu")
plt.xlabel('Recorte tempo')
if ("cmsGMM" in globals()):
    T.append("GMM-UBM")
    plt.plot(cmsGMM.vecBAS,cmsGMM.vecTPR,'s-.', linewidth=2)
if ("cmsiVEC" in globals()):
    T.append("i-vector")
    plt.plot(cmsiVEC.vecBAS,cmsiVEC.vecTPR,'s-.', linewidth=2)
if ("cmsxTDDNN" in globals()):
    T.append("v-vector TDDNN")
    plt.plot(cmsxTDDNN.vecBAS,cmsxTDDNN.vecTPR,'s-.', linewidth=2)    
if ("cmsxRESNET" in globals()):
    T.append("x-vector RESNET")
    plt.plot(cmsxRESNET.vecBAS,cmsxRESNET.vecTPR,'s-.', linewidth=2)
plt.grid(color='g', linestyle='-.', linewidth=0.5)
plt.legend(T)

T = []
fig = plt.figure(figsize =(9, 6))
plt.title("Taxas de verdadeiro negativo")
plt.ylabel("pu")
plt.xlabel('Recorte tempo')
if ("cmsGMM" in globals()):
    T.append("GMM-UBM")
    plt.plot(cmsGMM.vecBAS,cmsGMM.vecTNR,'s-.', linewidth=2)
if ("cmsiVEC" in globals()):
    T.append("i-vector")
    plt.plot(cmsiVEC.vecBAS,cmsiVEC.vecTNR,'s-.', linewidth=2)
if ("cmsxTDDNN" in globals()):
    T.append("v-vector TDDNN")
    plt.plot(cmsxTDDNN.vecBAS,cmsxTDDNN.vecTNR,'s-.', linewidth=2)    
if ("cmsxRESNET" in globals()):
    T.append("x-vector RESNET")
    plt.plot(cmsxRESNET.vecBAS,cmsxRESNET.vecTNR,'s-.', linewidth=2)
plt.grid(color='g', linestyle='-.', linewidth=0.5)
plt.legend(T)

T = []
fig = plt.figure(figsize =(9, 6))
plt.title("Acurácia balanceada")
plt.ylabel("pu")
plt.xlabel('Recorte tempo')
if ("cmsGMM" in globals()):
    T.append("GMM-UBM")
    plt.plot(cmsGMM.vecBAS,cmsGMM.vecBAC,'s-.', linewidth=2)
if ("cmsiVEC" in globals()):
    T.append("i-vector")
    plt.plot(cmsiVEC.vecBAS,cmsiVEC.vecBAC,'s-.', linewidth=2)
if ("cmsxTDDNN" in globals()):
    T.append("v-vector TDDNN")
    plt.plot(cmsxTDDNN.vecBAS,cmsxTDDNN.vecBAC,'s-.', linewidth=2)    
if ("cmsxRESNET" in globals()):
    T.append("x-vector RESNET")
    plt.plot(cmsxRESNET.vecBAS,cmsxRESNET.vecBAC,'s-.', linewidth=2)
plt.grid(color='g', linestyle='-.', linewidth=0.5)
plt.legend(T)    
# -----------------------------------------------------------------------------    