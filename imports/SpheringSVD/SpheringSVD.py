"""
Created on Fri Sep  3 15:03:54 2021

@author: adelino
"""
import numpy as np
# ---- POR ADELINO ------------------------------------------------------------
class SpheringSVD:
    def __init__(self):
        self.epsilon = 1e-5
        self.Mean = 0;
        self.Z = 1;
        
    def fit(self,X):
        self.Mean = np.mean(X, axis=0)
        Xc = X - self.Mean
        sigma = np.cov(Xc.T, rowvar=True) # [M x M]
        U,S,V = np.linalg.svd(sigma)
        # self.Z = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + self.epsilon)), U.T)) 
        self.Z = np.dot(V, np.diag(1.0/np.sqrt(S + self.epsilon)))
        
    def transform(self, X):
        Xc = (X - self.Mean)
        return np.matmul(Xc,self.Z)

# -----------------------------------------------------------------------------  