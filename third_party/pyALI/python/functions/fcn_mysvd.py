""" fcn_mysvd A simple routine for computing sigular value decomposition (SVD) with 'ncomp'
% singular values
%
% This function implements the approach described in Mukamel et. al. 2009 Neuron
% paper: "Automated analysis of cellular signals from large-scale calcium imaging data"
"""
import numpy as np
from scipy.sparse.linalg import eigs
def fcn_mysvd(data,ncomp):

    sz=data.shape
    
    if sz[1]<sz[0]:   
        cc=np.matmul(data.T,data)
        D,V=eigs(cc,ncomp)
        S=np.sqrt(np.diag(D))
        U=data@V@np.linalg.inv(S)
    else:
        cc=np.matmul(data,data.T)
        D,U=eigs(cc,ncomp)
        S=np.sqrt(np.diag(D))
        V=(np.linalg.inv(S)@U.T@data).T    
    
    return np.real(U),np.real(S),np.real(V)
    
    
