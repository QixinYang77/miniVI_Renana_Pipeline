import numpy as np

def ali_fp_support(footprint, clust_cen, r):
    """
    Set the footprint values to zeros beyond a 'r'-pixel circular
    region around the center.

    Parameters:
    footprint : numpy.ndarray
        nrow x ncol x nclust, footprints of ALI clusters
    clust_cen : numpy.ndarray
        nclust x 2 center coordinates of ALI clusters
    r : float
        radius of footprint support

    Returns:
    footprint : numpy.ndarray
        nrow x ncol x nclust, footprints with limited spatial support
    support : numpy.ndarray
        nrow x ncol x nclust, non-zero pixels of the footprint
    """
    nrow, ncol, nclust = footprint.shape

    for k in range(nclust):
        for i in range(nrow):
            for j in range(ncol):
                d = np.sqrt((i + 1 - clust_cen[0, k])**2 + (j + 1 - clust_cen[1, k])**2)
                if d > r:
                    footprint[i, j, k] = 0

    support = footprint != 0
    return footprint, support
