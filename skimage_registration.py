# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:07:53 2022

@author: andrea
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import cv2

folder = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Documenti\\PythonProjects\\CV2Apps\\Plant_analysis\\images'
path0 = folder + '\\WT_1_3_t0000_z0000_c0.tif'
path1 = folder + '\\WT_1_3_t0100_z0000_c0.tif'

image0 = cv2.imread(path0, -1)
image0 = np.float32(image0)
image0 = image0[1036:1116, 800:880] 


image1 = cv2.imread(path1, -1)
image1 = np.float32(image1)
image1 = image1[1036:1116, 800:880] 



vmin = np.amin(image0)
vmax = np.amax(image0)
image0 = (image0-vmin) / (vmax-vmin)
image0 = (image0*255).astype('uint8')
image1 = (image1-vmin) / (vmax-vmin)
image1 = (image1*255).astype('uint8')


# --- Compute the optical flow
v, u = optical_flow_tvl1(image0, image1)

# --- Use the estimated optical flow for registration

nr, nc = image0.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                     indexing='ij')

image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]),
                   mode='edge')


fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 10))

ax0.imshow(image0)
ax0.set_title("image0")
ax0.set_axis_off()

ax1.imshow(image1)
ax1.set_title("image1")
ax1.set_axis_off()

ax2.imshow(image1_warp)
ax2.set_title("registered")
ax2.set_axis_off()

fig.tight_layout()