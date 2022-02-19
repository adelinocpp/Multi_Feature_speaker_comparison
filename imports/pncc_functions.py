#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:53:07 2022

@author: adelino
"""
import numpy as np

# -----------------------------------------------------------------------------
class queue:
    def __init__(self,iM,iNumFilts):
        self.Queue_iWindow  = 2*iM + 1
        self.Queue_aad_P    = np.zeros((self.Queue_iWindow, iNumFilts))
        self.Queue_iHead    = 0
        self.Queue_iTail    = 0
        self.Queue_iNumElem = 0
    def queue_poll(self):
        if (self.Queue_iNumElem <= 0):
            print('Error: No elements')
            return 
        ad_x =  self.Queue_aad_P[self.Queue_iHead, :]
        self.Queue_iHead    = np.mod(self.Queue_iHead + 1, self.Queue_iWindow)       
        self.Queue_iNumElem = self.Queue_iNumElem - 1
        return ad_x
    
    def queue_avg(self,iNumFilts):
        adMean = np.zeros((iNumFilts,)) # Changed from 40 (number of filter banks)
        iPos = self.Queue_iHead;        
        for i in range(0,self.Queue_iNumElem):
            adMean = adMean + self.Queue_aad_P[iPos,: ]
            iPos   = np.mod(iPos + 1, self.Queue_iWindow);
        return adMean / self.Queue_iNumElem
    def queue_offer(self,ad_x):
        self.Queue_aad_P[self.Queue_iTail, :] = ad_x.T
        self.Queue_iTail    = np.mod(self.Queue_iTail + 1, self.Queue_iWindow)
        self.Queue_iNumElem = self.Queue_iNumElem + 1;
        if (self.Queue_iNumElem > self.Queue_iWindow):
            print('Error: Queue overflow') 