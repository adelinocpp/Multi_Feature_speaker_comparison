#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:51:14 2022

@author: adelino
"""
import config as c
import dill
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.mixture import GaussianMixture
# from scipy import stats

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, det_curve
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import simpson

def integrate_1D_GMM(gmm_model,xmin,xmax,npts=1000):
    if (xmax < xmin):
        xtemp = xmax
        xmax = xmin
        xmin = xmax
    xScore = np.linspace(xmin,xmax,npts)
    yDens = np.exp(gmm_model.score_samples(xScore.reshape(-1,1)))
    return simpson(yDens,xScore)

Analise_GMM_UBM = True
Analise_ivector = True
Analise_xvector_TDDNN = True
Analise_xvector_RESNET = True
plt.close('all')

ofile = open(c.DB_TIME_INFO_FILE, "rb")
dbTimeInfo = dill.load(ofile)
ofile.close()

if (Analise_GMM_UBM):
    if not os.path.exists(c.GMM_RESULT_PARAM):
        print("Calculando a calibracao gmm...")
        ofile = open(c.GMM_CALIBRATE_FILEDATA, "rb")
        xvector_calibrate_list = dill.load(ofile)
        ofile.close()
        
        vecScore = np.array([])
        vecResul = np.array([])
        vecSameSpeaker = np.array([])
        vecDiffSpeaker = np.array([])
        scoreMax = sys.float_info.min
        scoreMin = sys.float_info.max
        for idxL, calibrate in enumerate(xvector_calibrate_list):
            dataBSV = np.array(calibrate['speaker_BSV'])
            idxSameSpeaker = (np.array(calibrate['matchSpeaker']) == 1).nonzero()[0]
            idxDiffSpeaker = (np.array(calibrate['matchSpeaker']) == 0).nonzero()[0]
            if (scoreMax < dataBSV.max()):
                scoreMax = dataBSV.max()
            if (scoreMin > dataBSV.min()):
                scoreMin = dataBSV.min()
            vecSameSpeaker = np.hstack((vecSameSpeaker,dataBSV[idxSameSpeaker]))
            vecDiffSpeaker = np.hstack((vecDiffSpeaker,dataBSV[idxDiffSpeaker]))
            vecScore = np.hstack((vecScore,dataBSV))
            vecResul = np.hstack((vecResul,np.array(calibrate['matchSpeaker'])))
        
        # scaler = MinMaxScaler()
        # scaler.fit(vecScore)
        # vecScore = scaler.transform(vecScore)
        model = LogisticRegressionCV(solver='liblinear', random_state=0, class_weight='balanced')
        model.fit(vecScore.reshape(-1,1), vecResul)
        xvector_result = {}
        
        fpr, tpr, thresholds = roc_curve(vecResul, vecScore, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        x = np.linspace(scoreMin,scoreMax,2000)
        # kdeSS = stats.gaussian_kde(vecSameSpeaker)
        kdeSS = GaussianMixture(n_components=8)
        kdeSS.fit(vecSameSpeaker.reshape(-1, 1))
        # kdeDS = stats.gaussian_kde(vecDiffSpeaker)
        kdeDS = GaussianMixture(n_components=8)
        kdeDS.fit(vecDiffSpeaker.reshape(-1, 1))
        
        xvector_result['fpr'] = fpr
        xvector_result['tpr'] = tpr
        xvector_result['thresholds'] = thresholds
        xvector_result['thresh'] = thresh
        xvector_result['eer'] = eer
        xvector_result['x'] = x
        xvector_result['kdeSS'] = kdeSS
        xvector_result['kdeDS'] = kdeDS
        xvector_result['maxScore'] = scoreMax
        xvector_result['minScore'] = scoreMin
        xScore = np.linspace(scoreMin,scoreMax,1000)
        xvector_result['yPDFkdeSS'] = np.exp(kdeSS.score_samples(xScore.reshape(-1,1)))
        xvector_result['yPDFkdeDS'] = np.exp(kdeDS.score_samples(xScore.reshape(-1,1)))
        
        fpr, tpr, thresholds = det_curve(vecResul, vecScore, pos_label=1)
        xvector_result['det_fpr'] = fpr
        xvector_result['det_tpr'] = tpr
        xvector_result['det_thresholds'] = thresholds
        xvector_result['logistic_model'] = model
        
        
        ofile = open(c.GMM_RESULT_PARAM, "wb")
        dill.dump(xvector_result, ofile)
        ofile.close()
    else:
        print("Calibracao gmm carregada")
        ofile = open(c.GMM_RESULT_PARAM, "rb")
        xvector_result = dill.load(ofile)
        ofile.close()
    # validacao
    if (os.path.exists(c.GMM_VALIDATE_FILEDATA)):
        print("Validação gmm iniciada...")
        ofile = open(c.GMM_VALIDATE_FILEDATA, "rb")
        xvector_validate_list = dill.load(ofile)
        ofile.close()
        
        thr = xvector_result['thresh']
        kdeSS = xvector_result['kdeSS']
        kdeDS = xvector_result['kdeDS']
        log_model = xvector_result['logistic_model']
        
        validate_result = {}
        expResult = np.array([])
        scrResult = np.array([])
        thrResult = np.array([])
        logResult = np.array([])
        lgpResult = np.array([])
        corResult = np.array([])
        clgResult = np.array([])
        timResult = np.array([])
        
        for idxL, calibrate in enumerate(xvector_validate_list):
            dataBSV = np.array(calibrate['speaker_BSV'])
            dataWSV = np.array(calibrate['speaker_WSV'])
            nSpeaker = len(calibrate["speaker_BSV"])
            idxLoc = (np.array(calibrate["matchSpeaker"]) == 1).nonzero()[0][0]
            # idxSS = idxLoc*nSpeaker + idxLoc
            idxSS = idxLoc
            idxDS = np.array([x for x in range(0,nSpeaker) if (x != idxSS)])
            # listScoreBSV = calibrate["speaker_BSV"]
            
            scorePair = dataBSV[idxSS]
            dataBSV = dataBSV[idxDS]

            vecDATAS = np.hstack((dataBSV,dataWSV))
            minData = np.min(vecDATAS)
            maxData = np.max(vecDATAS)
            # kdeBSV = stats.gaussian_kde(dataBSV)
            kdeBSV = GaussianMixture(n_components=8)
            kdeBSV.fit(dataBSV.reshape(-1, 1))
            # kdeWSV = stats.gaussian_kde(dataWSV)
            kdeWSV = GaussianMixture(n_components=8)
            kdeWSV.fit(dataWSV.reshape(-1, 1))
            

            xScore = np.linspace(xvector_result['minScore'],xvector_result['maxScore'],1000)
            xvector_result['yPDFkdeSS']
            xvector_result['yPDFkdeDS']
            
            scoreSS = integrate_1D_GMM(kdeSS, minData, scorePair)
            scoreDS = integrate_1D_GMM(kdeDS, minData, scorePair)
            
            idxPair = np.where(dbTimeInfo.speaker_list==calibrate['Speaker_id'])[0][0]
            idxBSV = np.searchsorted(dbTimeInfo.speaker_list, np.array(calibrate["listBSVFileComp"])[idxDS])
            idxWSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listWSVFileComp"])
            kBay = scoreSS/scoreDS
            
            expResult = np.append(expResult,1)
            scrResult = np.append(scrResult,scorePair)
            thrResult = np.append(thrResult,int(scorePair > thr))
            logResult = np.append(logResult,log_model.predict(scorePair.reshape(1,-1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(scorePair.reshape(1,-1)))
            corResult = np.append(corResult,int(scorePair*kBay > thr))
            clgResult = np.append(clgResult,log_model.predict((scorePair*kBay).reshape(1,-1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxPair])
            
            expResult = np.append(expResult,np.zeros((len(idxBSV),)))
            scrResult = np.append(scrResult,dataBSV)
            thrResult = np.append(thrResult, np.array(dataBSV>thr,dtype=np.int32))
            logResult = np.append(logResult,log_model.predict(dataBSV.reshape(-1,1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(dataBSV.reshape(-1,1)))
            corResult = np.append(corResult,np.array(dataBSV*kBay > thr,dtype=np.int32))
            clgResult = np.append(clgResult,log_model.predict((dataBSV*kBay).reshape(-1,1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxBSV])
            
            expResult = np.append(expResult,np.zeros((len(idxWSV),)))
            scrResult = np.append(scrResult,dataWSV)
            thrResult = np.append(thrResult, np.array(dataWSV>thr,dtype=np.int32))
            logResult = np.append(logResult,log_model.predict(dataWSV.reshape(-1,1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(dataWSV.reshape(-1,1)))
            corResult = np.append(corResult,np.array(dataWSV*kBay > thr,dtype=np.int32))
            clgResult = np.append(clgResult,log_model.predict((dataWSV*kBay).reshape(-1,1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxWSV])
            
            print("Validação gmm {:} de {:}.".format(idxL,len(xvector_validate_list)-1))
            if (len(expResult) != len(scrResult)) and (len(thrResult) != len(logResult)) and \
               (len(lgpResult) != len(corResult)) and (len(clgResult) != len(timResult)):
                   print("Problema com tamanho dos vetores...")
            # sys.exit("Verificando...")
            
        validate_result['expected']     = expResult
        validate_result['score']        = scrResult
        validate_result['byThreshold']  = thrResult
        validate_result['byLogistic']   = logResult
        validate_result['byLogProb']    = lgpResult
        validate_result['byCorrection'] = corResult
        validate_result['byLogisCorr']  = clgResult
        validate_result['timeAudio']    = timResult
        ofile = open(c.GMM_VALIDATE_RESULT, "wb")
        dill.dump(validate_result, ofile)
        ofile.close();
        print("Validação gmm finalizada")
    
    
if (Analise_ivector):
    if not os.path.exists(c.IVECTOR_RESULT_PARAM):
        print("Calculando a calibracao ivector...")
        ofile = open(c.IVECTOR_CALIBRATE_FILEDATA, "rb")
        xvector_calibrate_list = dill.load(ofile)
        ofile.close()
        
        vecScore = np.array([])
        vecResul = np.array([])
        vecSameSpeaker = np.array([])
        vecDiffSpeaker = np.array([])
        scoreMax = sys.float_info.min
        scoreMin = sys.float_info.max
        for idxL, calibrate in enumerate(xvector_calibrate_list):
            dataBSV = np.array(calibrate['speaker_BSV'])
            idxSameSpeaker = (np.array(calibrate['matchSpeaker']) == 1).nonzero()[0]
            idxDiffSpeaker = (np.array(calibrate['matchSpeaker']) == 0).nonzero()[0]
            if (scoreMax < dataBSV.max()):
                scoreMax = dataBSV.max()
            if (scoreMin > dataBSV.min()):
                scoreMin = dataBSV.min()
            vecSameSpeaker = np.hstack((vecSameSpeaker,dataBSV[idxSameSpeaker]))
            vecDiffSpeaker = np.hstack((vecDiffSpeaker,dataBSV[idxDiffSpeaker]))
            vecScore = np.hstack((vecScore,dataBSV))
            vecResul = np.hstack((vecResul,np.array(calibrate['matchSpeaker'])))
        
        # scaler = MinMaxScaler()
        # scaler.fit(vecScore)
        # vecScore = scaler.transform(vecScore)
        model = LogisticRegressionCV(solver='liblinear', random_state=0, class_weight='balanced')
        model.fit(vecScore.reshape(-1,1), vecResul)
        xvector_result = {}
        
        fpr, tpr, thresholds = roc_curve(vecResul, vecScore, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        x = np.linspace(scoreMin,scoreMax,2000)
        # kdeSS = stats.gaussian_kde(vecSameSpeaker)
        kdeSS = GaussianMixture(n_components=8)
        kdeSS.fit(vecSameSpeaker.reshape(-1, 1))
        # kdeDS = stats.gaussian_kde(vecDiffSpeaker)
        kdeDS = GaussianMixture(n_components=8)
        kdeDS.fit(vecDiffSpeaker.reshape(-1, 1))
        
        xvector_result['fpr'] = fpr
        xvector_result['tpr'] = tpr
        xvector_result['thresholds'] = thresholds
        xvector_result['thresh'] = thresh
        xvector_result['eer'] = eer
        xvector_result['x'] = x
        xvector_result['kdeSS'] = kdeSS
        xvector_result['kdeDS'] = kdeDS
        xvector_result['maxScore'] = scoreMax
        xvector_result['minScore'] = scoreMin
        xScore = np.linspace(scoreMin,scoreMax,1000)
        xvector_result['yPDFkdeSS'] = np.exp(kdeSS.score_samples(xScore.reshape(-1,1)))
        xvector_result['yPDFkdeDS'] = np.exp(kdeDS.score_samples(xScore.reshape(-1,1)))
        
        fpr, tpr, thresholds = det_curve(vecResul, vecScore, pos_label=1)
        xvector_result['det_fpr'] = fpr
        xvector_result['det_tpr'] = tpr
        xvector_result['det_thresholds'] = thresholds
        xvector_result['logistic_model'] = model
        
        
        ofile = open(c.IVECTOR_RESULT_PARAM, "wb")
        dill.dump(xvector_result, ofile)
        ofile.close()
    else:
        print("Calibracao ivector carregada")
        ofile = open(c.IVECTOR_RESULT_PARAM, "rb")
        xvector_result = dill.load(ofile)
        ofile.close()
    # validacao
    if (os.path.exists(c.IVECTOR_VALIDATE_FILEDATA)):
        print("Validação ivector iniciada...")
        ofile = open(c.IVECTOR_VALIDATE_FILEDATA, "rb")
        xvector_validate_list = dill.load(ofile)
        ofile.close()
        
        thr = xvector_result['thresh']
        kdeSS = xvector_result['kdeSS']
        kdeDS = xvector_result['kdeDS']
        log_model = xvector_result['logistic_model']
        
        validate_result = {}
        expResult = np.array([])
        scrResult = np.array([])
        thrResult = np.array([])
        logResult = np.array([])
        lgpResult = np.array([])
        corResult = np.array([])
        clgResult = np.array([])
        timResult = np.array([])
        
        for idxL, calibrate in enumerate(xvector_validate_list):
            dataBSV = np.array(calibrate['speaker_BSV'])
            dataWSV = np.array(calibrate['speaker_WSV'])
            scorePair = calibrate["Pair_Score"]
            
            vecDATAS = np.hstack((dataBSV,dataWSV))
            minData = np.min(vecDATAS)
            maxData = np.max(vecDATAS)
            # kdeBSV = stats.gaussian_kde(dataBSV)
            kdeBSV = GaussianMixture(n_components=8)
            kdeWSV.fit(dataBSV.reshape(-1, 1))
            # kdeWSV = stats.gaussian_kde(dataWSV)
            kdeWSV = GaussianMixture(n_components=8)
            kdeWSV.fit(dataWSV.reshape(-1, 1))
            
            scoreSS = integrate_1D_GMM(kdeSS,minData, scorePair)
            scoreDS = integrate_1D_GMM(kdeDS,minData, scorePair)
            
            idxPair = np.where(dbTimeInfo.speaker_list==calibrate['Speaker_id'])[0][0]
            idxBSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listBSVFileComp"])
            idxWSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listWSVFileComp"])
            kBay = scoreSS/scoreDS
            
            expResult = np.append(expResult,1)
            scrResult = np.append(scrResult,scorePair)
            thrResult = np.append(thrResult,int(scorePair > thr))
            logResult = np.append(logResult,log_model.predict(scorePair.reshape(1,-1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(scorePair.reshape(1,-1)))
            corResult = np.append(corResult,int(scorePair*kBay > thr))
            clgResult = np.append(clgResult,log_model.predict((scorePair*kBay).reshape(1,-1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxPair])
            
            expResult = np.append(expResult,np.zeros((len(idxBSV),)))
            scrResult = np.append(scrResult,dataBSV)
            thrResult = np.append(thrResult, np.array(dataBSV>thr,dtype=np.int32))
            logResult = np.append(logResult,log_model.predict(dataBSV.reshape(-1,1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(dataBSV.reshape(-1,1)))
            corResult = np.append(corResult,np.array(dataBSV*kBay > thr,dtype=np.int32))
            clgResult = np.append(clgResult,log_model.predict((dataBSV*kBay).reshape(-1,1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxBSV])
            
            expResult = np.append(expResult,np.zeros((len(idxWSV),)))
            scrResult = np.append(scrResult,dataWSV)
            thrResult = np.append(thrResult, np.array(dataWSV>thr,dtype=np.int32))
            logResult = np.append(logResult,log_model.predict(dataWSV.reshape(-1,1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(dataWSV.reshape(-1,1)))
            corResult = np.append(corResult,np.array(dataWSV*kBay > thr,dtype=np.int32))
            clgResult = np.append(clgResult,log_model.predict((dataWSV*kBay).reshape(-1,1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxWSV])
            
            print("Validação ivector {:} de {:}.".format(idxL,len(xvector_validate_list)-1))
            if (len(expResult) != len(scrResult)) and (len(thrResult) != len(logResult)) and \
               (len(lgpResult) != len(corResult)) and (len(clgResult) != len(timResult)):
                   print("Problema com tamanho dos vetores...")
            # sys.exit("Verificando...")
            
        validate_result['expected']     = expResult
        validate_result['score']        = scrResult
        validate_result['byThreshold']  = thrResult
        validate_result['byLogistic']   = logResult
        validate_result['byLogProb']    = lgpResult
        validate_result['byCorrection'] = corResult
        validate_result['byLogisCorr']  = clgResult
        validate_result['timeAudio']    = timResult
        ofile = open(c.IVECTOR_VALIDATE_RESULT, "wb")
        dill.dump(validate_result, ofile)
        ofile.close();
        print("Validação ivector finalizada")


if (Analise_xvector_TDDNN):
   if not os.path.exists(c.XVECTOR_TDDNN_RESULT_PARAM):
       print("Calculando a calibracao TDDNN...")
       ofile = open(c.XVECTOR_TDDNN_CALIBRATE_FILEDATA, "rb")
       xvector_calibrate_list = dill.load(ofile)
       ofile.close()
       
       vecScore = np.array([])
       vecResul = np.array([])
       vecSameSpeaker = np.array([])
       vecDiffSpeaker = np.array([])
       scoreMax = sys.float_info.min
       scoreMin = sys.float_info.max
       for idxL, calibrate in enumerate(xvector_calibrate_list):
           dataBSV = np.array(calibrate['speaker_BSV'])
           idxSameSpeaker = (np.array(calibrate['matchSpeaker']) == 1).nonzero()[0]
           idxDiffSpeaker = (np.array(calibrate['matchSpeaker']) == 0).nonzero()[0]
           if (scoreMax < dataBSV.max()):
               scoreMax = dataBSV.max()
           if (scoreMin > dataBSV.min()):
               scoreMin = dataBSV.min()
           vecSameSpeaker = np.hstack((vecSameSpeaker,dataBSV[idxSameSpeaker]))
           vecDiffSpeaker = np.hstack((vecDiffSpeaker,dataBSV[idxDiffSpeaker]))
           vecScore = np.hstack((vecScore,dataBSV))
           vecResul = np.hstack((vecResul,np.array(calibrate['matchSpeaker'])))
       
       # scaler = MinMaxScaler()
       # scaler.fit(vecScore)
       # vecScore = scaler.transform(vecScore)
       model = LogisticRegressionCV(solver='liblinear', random_state=0, class_weight='balanced')
       model.fit(vecScore.reshape(-1,1), vecResul)
       xvector_result = {}
       
       fpr, tpr, thresholds = roc_curve(vecResul, vecScore, pos_label=1)
       eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
       thresh = interp1d(fpr, thresholds)(eer)
       x = np.linspace(scoreMin,scoreMax,2000)
       # kdeSS = stats.gaussian_kde(vecSameSpeaker)
       kdeSS = GaussianMixture(n_components=8)
       kdeSS.fit(vecSameSpeaker.reshape(-1, 1))
       # kdeDS = stats.gaussian_kde(vecDiffSpeaker)
       kdeDS = GaussianMixture(n_components=8)
       kdeDS.fit(vecDiffSpeaker.reshape(-1, 1))
       
       xvector_result['fpr'] = fpr
       xvector_result['tpr'] = tpr
       xvector_result['thresholds'] = thresholds
       xvector_result['thresh'] = thresh
       xvector_result['eer'] = eer
       xvector_result['x'] = x
       xvector_result['kdeSS'] = kdeSS
       xvector_result['kdeDS'] = kdeDS
       xvector_result['maxScore'] = scoreMax
       xvector_result['minScore'] = scoreMin
       xScore = np.linspace(scoreMin,scoreMax,1000)
       xvector_result['yPDFkdeSS'] = np.exp(kdeSS.score_samples(xScore.reshape(-1,1)))
       xvector_result['yPDFkdeDS'] = np.exp(kdeDS.score_samples(xScore.reshape(-1,1)))
       
       fpr, tpr, thresholds = det_curve(vecResul, vecScore, pos_label=1)
       xvector_result['det_fpr'] = fpr
       xvector_result['det_tpr'] = tpr
       xvector_result['det_thresholds'] = thresholds
       xvector_result['logistic_model'] = model
       
       
       ofile = open(c.XVECTOR_TDDNN_RESULT_PARAM, "wb")
       dill.dump(xvector_result, ofile)
       ofile.close()
   else:
       print("Calibracao TDDNN carregada")
       ofile = open(c.XVECTOR_TDDNN_RESULT_PARAM, "rb")
       xvector_result = dill.load(ofile)
       ofile.close()
   # validacao
   if (os.path.exists(c.XVECTOR_TDDNN_VALIDATE_FILEDATA)):
       print("Validação TDDNN iniciada...")
       ofile = open(c.XVECTOR_TDDNN_VALIDATE_FILEDATA, "rb")
       xvector_validate_list = dill.load(ofile)
       ofile.close()
       
       thr = xvector_result['thresh']
       kdeSS = xvector_result['kdeSS']
       kdeDS = xvector_result['kdeDS']
       log_model = xvector_result['logistic_model']
       
       validate_result = {}
       expResult = np.array([])
       scrResult = np.array([])
       thrResult = np.array([])
       logResult = np.array([])
       lgpResult = np.array([])
       corResult = np.array([])
       clgResult = np.array([])
       timResult = np.array([])
       
       for idxL, calibrate in enumerate(xvector_validate_list):
           dataBSV = np.array(calibrate['speaker_BSV'])
           dataWSV = np.array(calibrate['speaker_WSV'])
           scorePair = calibrate["Pair_Score"]
           
           vecDATAS = np.hstack((dataBSV,dataWSV))
           minData = np.min(vecDATAS)
           maxData = np.max(vecDATAS)
           # kdeBSV = stats.gaussian_kde(dataBSV)
           kdeBSV = GaussianMixture(n_components=8)
           kdeWSV.fit(dataBSV.reshape(-1, 1))
           # kdeWSV = stats.gaussian_kde(dataWSV)
           kdeWSV = GaussianMixture(n_components=8)
           kdeWSV.fit(dataWSV.reshape(-1, 1))
           
           scoreSS = integrate_1D_GMM(kdeSS, minData, scorePair)
           scoreDS = integrate_1D_GMM(kdeDS, minData, scorePair)
           
           idxPair = np.where(dbTimeInfo.speaker_list==calibrate['Speaker_id'])[0][0]
           idxBSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listBSVFileComp"])
           idxWSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listWSVFileComp"])
           kBay = scoreSS/scoreDS
           
           expResult = np.append(expResult,1)
           scrResult = np.append(scrResult,scorePair)
           thrResult = np.append(thrResult,int(scorePair > thr))
           logResult = np.append(logResult,log_model.predict(scorePair.reshape(1,-1)))
           lgpResult = np.append(lgpResult,log_model.predict_proba(scorePair.reshape(1,-1)))
           corResult = np.append(corResult,int(scorePair*kBay > thr))
           clgResult = np.append(clgResult,log_model.predict((scorePair*kBay).reshape(1,-1)))
           timResult = np.append(timResult,dbTimeInfo.speaker_time[idxPair])
           
           expResult = np.append(expResult,np.zeros((len(idxBSV),)))
           scrResult = np.append(scrResult,dataBSV)
           thrResult = np.append(thrResult, np.array(dataBSV>thr,dtype=np.int32))
           logResult = np.append(logResult,log_model.predict(dataBSV.reshape(-1,1)))
           lgpResult = np.append(lgpResult,log_model.predict_proba(dataBSV.reshape(-1,1)))
           corResult = np.append(corResult,np.array(dataBSV*kBay > thr,dtype=np.int32))
           clgResult = np.append(clgResult,log_model.predict((dataBSV*kBay).reshape(-1,1)))
           timResult = np.append(timResult,dbTimeInfo.speaker_time[idxBSV])
           
           expResult = np.append(expResult,np.zeros((len(idxWSV),)))
           scrResult = np.append(scrResult,dataWSV)
           thrResult = np.append(thrResult, np.array(dataWSV>thr,dtype=np.int32))
           logResult = np.append(logResult,log_model.predict(dataWSV.reshape(-1,1)))
           lgpResult = np.append(lgpResult,log_model.predict_proba(dataWSV.reshape(-1,1)))
           corResult = np.append(corResult,np.array(dataWSV*kBay > thr,dtype=np.int32))
           clgResult = np.append(clgResult,log_model.predict((dataWSV*kBay).reshape(-1,1)))
           timResult = np.append(timResult,dbTimeInfo.speaker_time[idxWSV])
           
           print("Validação TDDNN {:} de {}.".format(idxL,len(xvector_validate_list)-1))
           if (len(expResult) != len(scrResult)) and (len(thrResult) != len(logResult)) and \
              (len(lgpResult) != len(corResult)) and (len(clgResult) != len(timResult)):
                  print("Problema com tamanho dos vetores...")
           # sys.exit("Verificando...")
           
       validate_result['expected']     = expResult
       validate_result['score']        = scrResult
       validate_result['byThreshold']  = thrResult
       validate_result['byLogistic']   = logResult
       validate_result['byLogProb']    = lgpResult
       validate_result['byCorrection'] = corResult
       validate_result['byLogisCorr']  = clgResult
       validate_result['timeAudio']    = timResult
       ofile = open(c.XVECTOR_TDDNN_VALIDATE_RESULT, "wb")
       dill.dump(validate_result, ofile)
       ofile.close();
       print("Validação TDDNN finalizada")

if (Analise_xvector_RESNET):
    if not os.path.exists(c.XVECTOR_RESNET_RESULT_PARAM):
        print("Calculando a calibracao RESNET...")
        ofile = open(c.XVECTOR_RESNET_CALIBRATE_FILEDATA, "rb")
        xvector_calibrate_list = dill.load(ofile)
        ofile.close()
        
        vecScore = np.array([])
        vecResul = np.array([])
        vecSameSpeaker = np.array([])
        vecDiffSpeaker = np.array([])
        scoreMax = sys.float_info.min
        scoreMin = sys.float_info.max
        for idxL, calibrate in enumerate(xvector_calibrate_list):
            dataBSV = np.array(calibrate['speaker_BSV'])
            idxSameSpeaker = (np.array(calibrate['matchSpeaker']) == 1).nonzero()[0]
            idxDiffSpeaker = (np.array(calibrate['matchSpeaker']) == 0).nonzero()[0]
            if (scoreMax < dataBSV.max()):
                scoreMax = dataBSV.max()
            if (scoreMin > dataBSV.min()):
                scoreMin = dataBSV.min()
            vecSameSpeaker = np.hstack((vecSameSpeaker,dataBSV[idxSameSpeaker]))
            vecDiffSpeaker = np.hstack((vecDiffSpeaker,dataBSV[idxDiffSpeaker]))
            vecScore = np.hstack((vecScore,dataBSV))
            vecResul = np.hstack((vecResul,np.array(calibrate['matchSpeaker'])))
        
        # scaler = MinMaxScaler()
        # scaler.fit(vecScore)
        # vecScore = scaler.transform(vecScore)
        model = LogisticRegressionCV(solver='liblinear', random_state=0, class_weight='balanced')
        model.fit(vecScore.reshape(-1,1), vecResul)
        xvector_result = {}
        
        fpr, tpr, thresholds = roc_curve(vecResul, vecScore, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        x = np.linspace(scoreMin,scoreMax,2000)
        # kdeSS = stats.gaussian_kde(vecSameSpeaker)
        kdeSS = GaussianMixture(n_components=8)
        kdeSS.fit(vecSameSpeaker.reshape(-1, 1))
        # kdeDS = stats.gaussian_kde(vecDiffSpeaker)
        kdeDS = GaussianMixture(n_components=8)
        kdeDS.fit(vecDiffSpeaker.reshape(-1, 1))
        
        xvector_result['fpr'] = fpr
        xvector_result['tpr'] = tpr
        xvector_result['thresholds'] = thresholds
        xvector_result['thresh'] = thresh
        xvector_result['eer'] = eer
        xvector_result['x'] = x
        xvector_result['kdeSS'] = kdeSS
        xvector_result['kdeDS'] = kdeDS
        xvector_result['maxScore'] = scoreMax
        xvector_result['minScore'] = scoreMin
        xScore = np.linspace(scoreMin,scoreMax,1000)
        xvector_result['yPDFkdeSS'] = np.exp(kdeSS.score_samples(xScore.reshape(-1,1)))
        xvector_result['yPDFkdeDS'] = np.exp(kdeDS.score_samples(xScore.reshape(-1,1)))
        
        fpr, tpr, thresholds = det_curve(vecResul, vecScore, pos_label=1)
        xvector_result['det_fpr'] = fpr
        xvector_result['det_tpr'] = tpr
        xvector_result['det_thresholds'] = thresholds
        xvector_result['logistic_model'] = model
        
        
        ofile = open(c.XVECTOR_RESNET_RESULT_PARAM, "wb")
        dill.dump(xvector_result, ofile)
        ofile.close()
    else:
        print("Calibracao RESNET carregada")
        ofile = open(c.XVECTOR_RESNET_RESULT_PARAM, "rb")
        xvector_result = dill.load(ofile)
        ofile.close()
    # validacao
    if (os.path.exists(c.XVECTOR_RESNET_VALIDATE_FILEDATA)):
        print("Validação RESNET iniciada...")
        ofile = open(c.XVECTOR_RESNET_VALIDATE_FILEDATA, "rb")
        xvector_validate_list = dill.load(ofile)
        ofile.close()
        
        thr = xvector_result['thresh']
        kdeSS = xvector_result['kdeSS']
        kdeDS = xvector_result['kdeDS']
        log_model = xvector_result['logistic_model']
        
        validate_result = {}
        expResult = np.array([])
        scrResult = np.array([])
        thrResult = np.array([])
        logResult = np.array([])
        lgpResult = np.array([])
        corResult = np.array([])
        clgResult = np.array([])
        timResult = np.array([])
        
        for idxL, calibrate in enumerate(xvector_validate_list):
            dataBSV = np.array(calibrate['speaker_BSV'])
            dataWSV = np.array(calibrate['speaker_WSV'])
            scorePair = calibrate["Pair_Score"]
            
            vecDATAS = np.hstack((dataBSV,dataWSV))
            minData = np.min(vecDATAS)
            maxData = np.max(vecDATAS)
            # kdeBSV = stats.gaussian_kde(dataBSV)
            kdeBSV = GaussianMixture(n_components=8)
            kdeWSV.fit(dataBSV.reshape(-1, 1))
            # kdeWSV = stats.gaussian_kde(dataWSV)
            kdeWSV = GaussianMixture(n_components=8)
            kdeWSV.fit(dataWSV.reshape(-1, 1))
            
            scoreSS = integrate_1D_GMM(kdeSS,minData, scorePair)
            scoreDS = integrate_1D_GMM(kdeDS,minData, scorePair)
            
            idxPair = np.where(dbTimeInfo.speaker_list==calibrate['Speaker_id'])[0][0]
            idxBSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listBSVFileComp"])
            idxWSV = np.searchsorted(dbTimeInfo.speaker_list, calibrate["listWSVFileComp"])
            kBay = scoreSS/scoreDS
            
            expResult = np.append(expResult,1)
            scrResult = np.append(scrResult,scorePair)
            thrResult = np.append(scrResult,int(scorePair > thr))
            logResult = np.append(logResult,log_model.predict(scorePair.reshape(1,-1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(scorePair.reshape(1,-1)))
            corResult = np.append(corResult,int(scorePair*kBay > thr))
            clgResult = np.append(clgResult,log_model.predict((scorePair*kBay).reshape(1,-1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxPair])
            
            expResult = np.append(expResult,np.zeros((len(idxBSV),)))
            scrResult = np.append(scrResult,dataBSV)
            thrResult = np.append(scrResult, np.array(dataBSV>thr,dtype=np.int32))
            logResult = np.append(logResult,log_model.predict(dataBSV.reshape(-1,1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(dataBSV.reshape(-1,1)))
            corResult = np.append(corResult,np.array(dataBSV*kBay > thr,dtype=np.int32))
            clgResult = np.append(clgResult,log_model.predict((dataBSV*kBay).reshape(-1,1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxBSV])
            
            expResult = np.append(expResult,np.zeros((len(idxWSV),)))
            scrResult = np.append(scrResult,dataWSV)
            thrResult = np.append(scrResult, np.array(dataWSV>thr,dtype=np.int32))
            logResult = np.append(logResult,log_model.predict(dataWSV.reshape(-1,1)))
            lgpResult = np.append(lgpResult,log_model.predict_proba(dataWSV.reshape(-1,1)))
            corResult = np.append(corResult,np.array(dataWSV*kBay > thr,dtype=np.int32))
            clgResult = np.append(clgResult,log_model.predict((dataWSV*kBay).reshape(-1,1)))
            timResult = np.append(timResult,dbTimeInfo.speaker_time[idxWSV])
            
            print("Validação RESNET {:} de {}.".format(idxL,len(xvector_validate_list)-1))
            if (len(expResult) != len(scrResult)) and (len(thrResult) != len(logResult)) and \
               (len(lgpResult) != len(corResult)) and (len(clgResult) != len(timResult)):
                   print("Problema com tamanho dos vetores...")
            # sys.exit("Verificando...")
            
        validate_result['expected']     = expResult
        validate_result['score']        = scrResult
        validate_result['byThreshold']  = thrResult
        validate_result['byLogistic']   = logResult
        validate_result['byLogProb']    = lgpResult
        validate_result['byCorrection'] = corResult
        validate_result['byLogisCorr']  = clgResult
        validate_result['timeAudio']    = timResult
        ofile = open(c.XVECTOR_RESNET_VALIDATE_RESULT, "wb")
        dill.dump(validate_result, ofile)
        ofile.close();
        print("Validação RESNET finalizada")
        
