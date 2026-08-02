import numpy as np

def ali_assign_cluster(sloc, clust_cen, radius):
    """
    Assign spikes to the nearest cluster

    if a spike is more than 'radius' pixels away from all the clusters, then it is not
    assigned to any cluster, ie., clust_idx=0

    input--
      sloc: nspike x 2, array of spike locations
      clust_cen: nclust x 2, center coordinates of ALI clusters

    output--
      clust_idx: nspike x 1, cluster index of each spike, values could range from 0 to nclust 
                                0 means that the spike is NOT assigned to any cluster         
    """
    nspike = sloc.shape[0]
    clust_idx = np.zeros(nspike, dtype=int)

    for i in range(nspike):
        dist = np.sqrt((sloc[i, 0] - clust_cen[0, :])**2 + (sloc[i, 1] - clust_cen[1, :])**2)
        m = np.min(dist)
        idx = np.argmin(dist) + 1  # +1 to have start with cluster #1
        if m < radius:
            clust_idx[i] = idx

    return clust_idx
