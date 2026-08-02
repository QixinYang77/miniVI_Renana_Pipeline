"""% ali_select_connected: Return a list of N brightest pixels in the input
 image that are connected to and within 'radius' pixels away from the starting
 pixel 'startpix'

 input--
   im: [nrow x ncol] input image 
   startpix: 2x1 array containing the row and column coordinates of the starting pixel
   N: scalar specify the number of pixels to be detected
   radius: maximal distance from the starting pixel, for a pixel to be detected

 output--
   pixel_list: 1x N array containing linear indices of N detected pixels
   roimap: [nrow x ncol] binary image of detected pixel location
   indexIJ: 2 x N array containing row and col indices of the N detected pixels 

"""
import numpy as np
def ali_select_connected(im,startpix,N,radius):

    ss=im.shape
    queue = np.copy(startpix[:,np.newaxis])
    queueval = im[queue[0],queue[1]]
    roimap = np.zeros(ss, dtype=bool)
    queuemap = np.zeros(ss)
    queuemap[queue[0],queue[1]]=1;
    indexIJ = None
    nsel=0;
    
    while nsel<N:
        # add brightest (i.e. first) pixel in queue to roi
        pix = queue[:,0,np.newaxis]
        roimap[pix[0],pix[1]]=1
        if indexIJ is None:
            indexIJ = np.copy(pix)
        else:
            indexIJ = np.append(indexIJ,pix, axis=1)
        queue = np.delete(queue,0,1)
        queueval = np.delete(queueval,0,0)
        nsel += 1
        # add pixels connected to the added pixel to queue
        candidate = np.asanyarray([[pix[0,0]],[pix[1,0]]]) + np.asarray([[1,-1,0,0],[0,0,1,-1]])
        d = np.sqrt(np.sum((candidate-startpix[:,np.newaxis])**2,0))
        idx = (candidate[0,:]>=0) & (candidate[0,:]<ss[0]) & (candidate[1,:]>=0) & (candidate[1,:]<ss[1]) & (d<=radius)
        candidate = candidate[:,idx]
        for i in range(candidate.shape[1]):
            pix = candidate[:,i, np.newaxis]
            if queuemap[pix[0],pix[1]]==0:
                queue = np.append(queue,pix, axis=1)
                queueval = np.append(queueval,im[pix[0],pix[1]])
                queuemap[pix[0],pix[1]]=1;
        # sort all pixels in queue
        idx = queueval[::-1].argsort()
        queue=queue[:,idx]
        queueval=queueval[idx]
    pixel_list = np.ravel_multi_index((indexIJ[0,:],indexIJ[1,:]),ss)
    
    return pixel_list,roimap,indexIJ
