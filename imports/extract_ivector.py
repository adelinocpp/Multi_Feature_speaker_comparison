#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 08:34:54 2021

@author: adelino
"""
import numpy as np

def extract_ivector(statVector, ubmData, T_matrix):
# extracts i-vector for stats in statFilename with UBM and T matrix in 
# ubmFilename and tvFilename, and optionally save the i-vector in ivFilename. 
#
# Inputs:
#   - statVector    : concatenated stats in a one-dimensional array
#   - ubmData       : structure with UBM hyperparameters
#   - T_matrix      : matriz de variabildiade total
#
# Outputs:
#   - x             : output identity vector (i-vector)  
#
# References:
#   [1] D. Matrouf, N. Scheffer, B. Fauve, J.-F. Bonastre, "A straightforward 
#       and efficient implementation of the factor analysis model for speaker 
#       verification," in Proc. INTERSPEECH, Antwerp, Belgium, Aug. 2007, 
#       pp. 1242-1245.  
#   [2] P. Kenny, "A small footprint i-vector extractor," in Proc. Odyssey, 
#       The Speaker and Language Recognition Workshop, Singapore, Jun. 2012.
#   [3] N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-end 
#       factor analysis for speaker verification," IEEE TASLP, vol. 19, pp. 
#       788-798, May 2011. 
#
#
# Omid Sadjadi <s.omid.sadjadi@gmail.com>
# MicroSoft Research, Silicon Valley Center

    nmix, ndim  = ubmData.means_.shape
    S = np.reshape(ubmData.covariances_.T, (ndim * nmix, 1))
    idx_sv = np.reshape(np.matlib.repmat(range(0,nmix), ndim, 1).T, (ndim * nmix, 1))


    tv_dim = T_matrix.shape[0]
    I = np.eye(tv_dim)
    T_invS =  np.divide(T_matrix, np.matlib.repmat(S.T,tv_dim,1))
    # T_invS =  bsxfun(@rdivide, T, S');
    N = statVector[:nmix]
    F = statVector[nmix:]

    L = I +  np.matmul(np.multiply(T_invS, N[idx_sv].T), T_matrix.T)
    B = np.matmul( T_invS,F)
    # B.shape = (B.shape[0],1)
    x = np.matmul(np.linalg.pinv(L),B)
    # L = I +  bsxfun(@times, T_invS, N(idx_sv)') * T';
    # B = T_invS * F;
    # x = pinv(L) * B;

    return x