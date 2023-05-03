#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:03:32 2021

@author: adelino
"""
import numpy as np
import numpy.matlib
def compute_bw_stats(vX, GMM):

# extracts sufficient statistics for features in feaFilename and GMM 
# ubmFilename, and optionally save the stats in statsFilename. The 
# first order statistics are centered.
#
# Inputs:
#   - feaFilename  : input feature file name (string) or a feature matrix 
#					(one observation per column)
#   - ubmFilename  : file name of the UBM or a structure with UBM 
#					 hyperparameters.
#   - statFilename : output file name (optional)   
#
# Outputs:
#   - N			   : mixture occupation counts (responsibilities) 
#   - F            : centered first order stats
#
# References:
#   [1] N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-end 
#       factor analysis for speaker verification," IEEE TASLP, vol. 19, pp. 788-798,
#       May 2011. 
#   [2] P. Kenny, "A small footprint i-vector extractor," in Proc. Odyssey, 
#       The Speaker and Language Recognition Workshop, Jun. 2012.
#
#
# Omid Sadjadi <s.omid.sadjadi@gmail.com>
# Microsoft Research, Conversational Systems Research Center

    nmix, ndim = GMM.means_.shape
    m = np.reshape(GMM.means_.T, (ndim * nmix, 1))
    idx_sv = np.reshape(np.matlib.repmat(range(0,nmix), ndim, 1).T, (ndim * nmix, 1))
    
    posterioriX = GMM.predict_proba(vX) + 1e-18*np.random.rand(vX.shape[0],nmix)
    N = posterioriX.sum(0)
    # F = np.matmul(posterioriX.T,vX)
    F = np.matmul(vX.T,posterioriX)
    # F = np.matmul(posterioriX.T,vX)
    
    F = np.reshape(F.T, (ndim * nmix, 1))
    F = F - np.multiply(N[idx_sv],m) # centered first order stats

    return N, F
# -----------------------------------------------------------------------------

