#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:04:15 2021

@author: adelino
"""
import numpy as np
import time
import numpy.linalg
import pickle
import os

def train_tv_space(data, UBM, tv_dim, nIter, filesave):
# uses statistics in dataLits to train the i-vector extractor with tv_dim 
# factors and niter EM iterations. The training process can be parallelized
# via parfor with nworkers. The output can be optionally saved in tvFilename.
#
# Technically, assuming a factor analysis (FA) model of the from:
#
#           M = m + T . x
#
# for mean supervectors M, the code computes the maximum likelihood 
# estimate (MLE)of the factor loading matrix T (aka the total variability 
# subspace). Here, M is the adapted mean supervector, m is the UBM mean 
# supervector, and x~N(0,I) is a vector of total factors (aka i-vector).
#
# Inputs:
#   - dataList    : ASCII file containing stats file names (1 file per line)
#                   or a cell array of concatenated stats (i.e., [N; F])
#   - ubmFilename : UBM file name or a structure with UBM hyperparameters
#   - tv_dim      : dimensionality of the total variability subspace
#   - niter       : number of EM iterations for total subspace learning
#   - nworkers    : number of parallel workers 
#   - tvFilename  : output total variability matrix file name (optional)
#
# Outputs:
#   - T 		  : total variability subspace matrix  
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
# Microsoft Research, Conversational Systems Research Center
    randInit = False
    epsMIN = 1e-3
    epsMAX = 1e4
    epsChange = 1e-5
    notChangeIter = 0
    maxNotChange = 10
    nmix, ndim = UBM.means_.shape
    S = np.reshape(UBM.covariances_, (ndim * nmix, 1));
    iniIter = 0
    if (os.path.exists(filesave)):
        with open(filesave, 'rb') as f:
            T = pickle.load(f)
            iniIter = pickle.load(f)
            iterMeanAnt = pickle.load(f)
            iterStdAnt = pickle.load(f)
            notChangeIter = pickle.load(f)
    else:
        print('\n\nRandomly initializing T matrix ...\n\n');
        randInit = True
        iterMeanAnt = 1e3
        iterStdAnt = 1e3
        # suggested in JFA cookbook
        # T = np.random.rand(tv_dim, ndim * nmix) * S.mean() * 0.01;
        T = np.random.rand(tv_dim, ndim * nmix) * S.sum(0) * 0.001;
        #T = randn(tv_dim, ndim * nmix) * mean(S);
    
    N, F = load_data(data, ndim, nmix)
        
    print('Re-estimating the total subspace with {:} factors ...\n'.format(tv_dim));
    for iter in range(iniIter,nIter):
        print('EM iter#: {:} \t'.format(iter))
        tim = time.time()
        LU, RU = expectation_tv(T, N, F, S, tv_dim, nmix, ndim);
        Tnew = maximization_tv(LU, RU, ndim, nmix);
        iterMean = np.absolute(Tnew - T).mean()
        iterStd = np.absolute(Tnew - T).std()
        tim = (time.time() - tim);
        print('[elaps = {:.2f} s, mean: {:3.2e}, std: {:3.2e}, notChange {:}]\n'.format(tim, iterMean, iterStd,notChangeIter))
        T = Tnew
        
        # --- CRITERIOS DE CONVERGENCIA ----------------------------------------
        if (((not randInit) and (iterMean < epsMIN) and (iterStd < epsMAX)) or (notChangeIter >= 10)):
            print('Estimantion convergence')
            with open(filesave, 'wb') as f:
                pickle.dump(T,f)
                pickle.dump(iter,f)
                pickle.dump(iterMeanAnt,f)
                pickle.dump(iterStdAnt,f)
                pickle.dump(notChangeIter,f)
            break
        if ( (np.absolute(iterMean - iterMeanAnt) < epsChange) and (np.absolute(iterStd - iterStdAnt) < epsChange) ):
            notChangeIter += 1
        else:
            notChangeIter = 0
        if ((not randInit) and (iterStd > epsMAX)):
            print('Estimantion divergence. Randomly re-initializing T matrix')
            randInit = True
            T = np.random.rand(tv_dim, ndim * nmix) * S.sum(0) * 0.001;
            iterMeanAnt = 1e3
            iterStdAnt = 1e3
            notChangeIter = 0
        if (randInit):
            randInit = False
            
        iterMeanAnt = iterMean
        iterStdAnt = iterStd
        with open(filesave, 'wb') as f:
            pickle.dump(T,f)
            pickle.dump(iter,f)
            pickle.dump(iterMeanAnt,f)
            pickle.dump(iterStdAnt,f)
            pickle.dump(notChangeIter,f)
        
    return T
# -----------------------------------------------------------------------------
def load_data(datalist, ndim, nmix):
    nfiles = datalist.shape[0]
    N = np.zeros((nfiles, nmix))
    F = np.zeros((nfiles, ndim * nmix))
    for file in range(0,nfiles):
        N[file, :] = datalist[file,:nmix]
        F[file, :] = datalist[file,nmix:]
    
    return N, F
# -----------------------------------------------------------------------------
def expectation_tv(T, N, F, S, tv_dim, nmix, ndim):
# compute the posterior means and covariance matrices of the factors 
# or latent variables
    idx_sv = np.reshape(np.matlib.repmat(range(0,nmix), ndim, 1).T, (ndim * nmix, 1))
    nfiles = F.shape[0];

    LU = np.zeros((nmix, tv_dim, tv_dim))
    RU = np.zeros((tv_dim, nmix * ndim))
    I = np.eye(tv_dim)
    # T_invS =  np.divide(T, S.T)
    T_invS =  np.divide(T, np.matlib.repmat(S.T,tv_dim,1))

    parts = 250; # modify this based on your resources
    nbatch = int(np.floor( nfiles/parts + 0.999999 ))
    for batch in range(0,nbatch):
        start = 0 + batch * parts
        fin = min((batch +1) * parts, nfiles)
        len = fin - start;
        index = range(start,fin)
        N1 = N[index, :]
        F1 = F[index, :]
        Ex = np.zeros((tv_dim, len))
        Exx = np.zeros((tv_dim, tv_dim, len))
        for ix in range(0,len):
            L = I +  np.matmul(np.multiply(T_invS, N1[ix, idx_sv].T), T.T)
            Cxx = np.linalg.pinv(L) # this is the posterior covariance Cov(x,x)
            # B = np.matmul( T_invS,F1[ix, :])
            B = np.matmul( T_invS,F1[ix, :])
            B.shape = (tv_dim,1)
            ExV = np.matmul(Cxx ,B) # this is the posterior mean E[x]
            ExV.shape = (tv_dim,)
            Ex[:, ix] = ExV
            # ExV = Ex[:, ix]
            ExV.shape = (tv_dim,1)
            # Exx[:, :, ix] = Cxx + np.matmul(Ex[:, ix],Ex[:, ix].T)
            Exx[:, :, ix] = Cxx + np.matmul(ExV,ExV.T)
            
        RU = RU + np.matmul(Ex,F1)
        for mix in range(0,nmix):
            # tmp = np.multiply(Exx, np.reshape(N1[:, mix].T,(1,1,len)))
            tmp = np.multiply(Exx, N1[:, mix])
            LU[mix,:,:] = LU[mix,:,:] + tmp.sum(2);
    return LU, RU
# -----------------------------------------------------------------------------
def maximization_tv(LU, RU, ndim, nmix):
# ML re-estimation of the total subspace matrix or the factor loading
# matrix
    for mix in range(0,nmix):
        idx = range( mix* ndim, (mix+1)* ndim);
        # print('mix = {:}'.format(mix))
        RU[:, idx] = np.linalg.solve(LU[mix,:,:],RU[:, idx])

    return RU