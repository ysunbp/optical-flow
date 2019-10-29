from skimage.io import imread
from scipy import signal,ndimage
import numpy as np
import time
import scipy.io as sio
from matplotlib.pyplot import imshow,show,figure
import skimage.transform as tf
import IPython
import flow_vis

image1 = imread('frame1.png',as_gray=True)
image2 = imread('frame2.png',as_gray=True)

flow_gt = sio.loadmat('flow_gt.mat')['groundTruth']
flow_image_gt = flow_vis.flow_to_color(flow_gt)


#PARTA:Lukas_kanade
def lukas_kanade(I1, I2, window_size=5):
    
    w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1 = I1/255.  # normalize pixels
    I2 = I2/255.   # normalize pixels
    
    # Define convolution kernels.
    kernel_x = [[-1/4,1/4],[-1/4,1/4]]
    kernel_y = [[-1/4,-1/4],[1/4,1/4]]
    kernel_t = [[1/4,1/4],[1/4,1/4]]
    
    # Compute partial derivatives.
    Ix = signal.convolve2d(I1+I2, kernel_x, mode = 'same')
    Iy = signal.convolve2d(I1+I2, kernel_y, mode = 'same')
    It = signal.convolve2d(I2-I1, kernel_t, mode = 'same')
    
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    for i in range(w, I1.shape[0] - w):
        for j in range(w, I1.shape[1] - w):

            # obtain partial derivatives for current patch
            px = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            py = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            pt = It[i - w:i + w + 1, j - w:j + w + 1].flatten()
            
        
            
            # Compute optical flow.
            A = np.column_stack((px, py))
            pt = np.array(pt)[np.newaxis]
            nu = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T,(-pt.T)))
            
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return u,v
    
    
h, w = image1.shape[:2]
u, v = lukas_kanade(image1, image2, )
current_y, current_x = np.mgrid[0:h, 0:w]
for i in range(1,10):
    flow_u = u*i
    flow_v = v*i
    
    image1_warp = ndimage.map_coordinates(image1, [current_y+flow_v, current_x+flow_u])
    
    imshow(image1_warp,cmap='gray')
    
    IPython.display.clear_output(True)
    
    show()
