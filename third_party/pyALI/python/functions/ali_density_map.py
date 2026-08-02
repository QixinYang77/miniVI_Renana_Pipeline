import numpy as np

def ali_density_map(sloc, sz, factor):
    # ali_density_map Generate spike density map (or ALI map) using 2D histogram
    #
    # input--
    #   sloc: nspike x 2 array, row and col coordinate of spikes
    #   sz:   [row, col] size of the raw image
    #   factor: ALI map contains factor times more pixel in both row,col dimensions
    # output--
    #   cnt: (row*factor) x (col*factor) array of spike count in each bin
    #   cen: list of 2 arrays, center locations of each bin

    bin_edges1 = np.linspace(0.5, sz[0] + 0.5, sz[0] * factor + 1)
    bin_edges2 = np.linspace(0.5, sz[1] + 0.5, sz[1] * factor + 1)

    cnt, edges1, edges2 = np.histogram2d(sloc[:, 0], sloc[:, 1], bins=[bin_edges1, bin_edges2])

    cnt = cnt[:-1, :-1]
    cen1 = (edges1[:-1] + edges1[1:]) / 2
    cen2 = (edges2[:-1] + edges2[1:]) / 2

    cen = [cen1, cen2]

    return cnt, cen
