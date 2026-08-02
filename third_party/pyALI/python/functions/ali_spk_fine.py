"""
ali_spk_fine Localize AP with sub-pixel precision through weighted averaging

input--
  df: nrow x ncol x nspike, stack of df_AP images, contain positively going spikes
  npix: scalar, number of pixels around the peak for calculating the average
  radius: scalar
  initloc: nspike x 2, coarse coordinates [row,col] of action potentials, integer pixel values
output--
  sloc: nspike x 2 array
        sloc[i,0] stores the row location of the i th spike;
        sloc[i,1] stores the col location of the i th spike
  bns: 1 x nspike, mean brightness of the region included in the average
  roi: nrow x ncol x nspike, BW images of the region included in the average
"""
import numpy as np
from ali_select_connected import ali_select_connected

def ali_spk_fine(df, npix, radius, initloc=None):


    sloc = []
    bns = []
    roi = []

    nspike = df.shape[2]
    for i in range(nspike):
        im = df[:, :, i]
        if initloc is None:
            M = np.max(im)
            I, J = np.where(im == M)
            # If multiple max, take the first
            I, J = I[0], J[0]
        else:
            I, J = initloc[i, 0], initloc[i, 1]

        pixel_list, roimap, indexIJ = ali_select_connected(im, np.array([I, J]), npix, radius)
        val = im.ravel()[pixel_list]
        w = val ** 2
        loc = (indexIJ * w[:,np.newaxis].T) / np.sum(w)
        loc = np.sum(loc, axis=1)# weighted avg
        bns.append(np.mean(val))
        sloc.append(loc)
        roi.append(roimap)

    sloc = np.array(sloc)
    bns = np.array(bns)
    roi = np.stack(roi, axis=2)

    return sloc, bns, roi
