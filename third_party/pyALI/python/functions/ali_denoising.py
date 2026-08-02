"""
ali_denoising Reduce noise by PCA/SVD, keeping nsvd components

input--
  df:   nrow x ncol x nframe, stack of df images, contain positively going spikes
  nsvd: number of svd components to keep
output--
  df_dnoised:   nrow x ncol x nframe, de-noised df images
  ucomps: nrow x ncol x nsvd kept spatial components
  vcomps: nframe x nsvd kept temporal components
"""
import numpy as np
from fcn_mysvd import fcn_mysvd

def ali_denoising(df, nsvd):
    sz = df.shape
    df_reshaped = df.reshape(-1, sz[2])
    
    uu, s, vv = fcn_mysvd(df_reshaped, nsvd)
    rec = uu @ s @ vv.T
    
    df_dnoised = rec.reshape(sz)
    ucomps = uu.reshape(sz[0], sz[1], nsvd)
    vcomps = vv.reshape(sz[2], nsvd)
    
    return df_dnoised, ucomps, vcomps
