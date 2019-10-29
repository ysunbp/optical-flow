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

num_layers = 1
downscale = 2

# Construct image pyramids
pyramids1 = tf.pyramid_gaussian(image1, max_layer=num_layers, downscale=downscale, sigma=1)
pyramids2 = tf.pyramid_gaussian(image2, max_layer=num_layers, downscale=downscale, sigma=1)

u = np.zeros(image1.shape)
v = np.zeros(image1.shape)
for im1,im2 in zip(reversed(list(pyramids1)),reversed(list(pyramids2))):

    # Upsampling and upscaling current flow estimation.
    h,w = im1.shape
    u = h/u.shape[0] * tf.resize(u, (h,w), order=1)
    v = h/u.shape[0] * tf.resize(v, (h,w), order=1)

    # Warp image.
    current_y, current_x = np.mgrid[0:h, 0:w]
    #print(np.shape(current_x))
    im1_warp = ndimage.map_coordinates(im1, [current_y+v,current_x+u])

    # Update optical flow.
    u_temp,v_temp = lukas_kanade(im1_warp, im2, )
    u = u+u_temp
    v = v+v_temp
    
    
    flow_image = flow_vis.flow_to_color(np.stack([u,v],axis=-1))
    figure()
    imshow(flow_image,cmap='gray')
