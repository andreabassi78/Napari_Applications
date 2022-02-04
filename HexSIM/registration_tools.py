# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:13:17 2022

@author: andrea
"""
import numpy as np


import cv2

def stack_registration(stack, z_idx, c_idx = 0, method = 'cv2', mode = 'Euclidean'):
    '''
    Stack registration works on 3D 4D stacks
    Paramenters:
    z_idx: index of the stack where the registration starts
    c_idx: for 4D stacks it is the channel on which the registration is performed. 
        The other channels are registered with the warp matrix found in this channel
    method: choose between 'optical flow', 'crosscorrelation', 'cv2'
    mode: type of registration for cv2
        
    Returns a 3D or 4D resistered stack
    
    '''
    
    
    def phase_cross_reg(ref, im):
        from scipy.ndimage import fourier_shift
        from skimage.registration import phase_cross_correlation
        
        shift, error, diffphase = phase_cross_correlation(ref,im)
        print(shift)
        reg = fourier_shift(np.fft.fftn(im), shift)
        reg = np.fft.ifftn(reg).real 
        return reg, None
    
    def optical_flow_reg(ref,im):  # --- Compute the optical flow
        from skimage.registration import optical_flow_tvl1 
        from skimage.transform import warp
        v, u = optical_flow_tvl1(ref, im)
        nr, nc = ref.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                             indexing='ij')
        reg = warp(im, np.array([row_coords + v, col_coords + u]))
        return reg, None
    
    def cv2_reg(ref,im, mode = mode):

        warp_mode_dct = {'Translation' : cv2.MOTION_TRANSLATION,
                         'Affine' : cv2.MOTION_AFFINE,
                         'Euclidean' : cv2.MOTION_EUCLIDEAN,
                         'Homography' : cv2.MOTION_HOMOGRAPHY
                         }
        warp_mode = warp_mode_dct[mode] 
        
        
        
        number_of_iterations = 3000
        termination_eps = 1e-6
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        number_of_iterations,  termination_eps)
            
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
        try:
            _, warp_matrix = cv2.findTransformECC((ref.astype(np.float32)), im.astype(np.float32),
                                                      warp_matrix, warp_mode, criteria)
            
            reg =  apply_warp(im, warp_matrix)
        except Exception as e:
            # print(f'{e}, frame not registered')
            reg = im
        
        return reg, warp_matrix 
    
    
    def apply_warp(im,w):
        sy,sx = im.shape
        reg = cv2.warpAffine(im, w, (sx,sy),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return reg
    
         
    if method == 'optical flow':
        image_registration = optical_flow_reg
    elif method == 'crosscorrelation':
        image_registration = phase_cross_reg
    elif method == 'cv2':
        image_registration = cv2_reg 
    else:
        raise(TypeError, f'registration method {method} not supported')
   
    s = stack.shape
    
    if stack.ndim ==4:
        sc = s[0]
        sz = s[1] 
        
        registered = np.zeros_like(stack)
        registered[c_idx,z_idx,:,:] = stack[c_idx,z_idx,:,:] 
        #otherchannels = [ci for ci in range(sc) if ci !=c_idx ] 
        #print(otherchannels)
        
        #register forwards
        for zi in range(z_idx,sz):
            ref_image = registered[c_idx,zi-1,:,:]
            current_image = stack[c_idx,zi,:,:]
            _, wm  = image_registration(ref_image, current_image)
            for ci in range(sc):
                otherimage = stack[ci,zi,:,:]
                registered[ci,zi,:,:] = apply_warp(otherimage, wm)      
        #register backwards
        for zi in range(z_idx-1,-1,-1):
            ref_image = registered[c_idx,zi+1,:,:]
            current_image = stack[c_idx,zi,:,:]
            _, wm  = image_registration(ref_image, current_image)
            for ci in range(sc):
                otherimage = stack[ci,zi,:,:]
                registered[ci,zi,:,:] = apply_warp(otherimage, wm)
        return registered    
         
    
    elif stack.ndim ==3:
        sz = s[0]
        registered = np.zeros_like(stack)
        registered[z_idx,:,:] = stack[z_idx,:,:]
        #register forwards
        for zi in range(z_idx+1,sz):
            ref_image = registered[zi-1,:,:]
            current_image = stack[zi,:,:]
            registered[zi,:,:],_ = image_registration(ref_image, current_image)  
        #register backwards
        for zi in range(z_idx-1,-1,-1):
            ref_image = registered[zi+1,:,:]
            current_image = stack[zi,:,:]
            registered[zi,:,:],_ = image_registration(ref_image, current_image)  
        return registered

    else:
        raise(TypeError( 'only 3D (z,y,x) or 4D (c,z,y,x) registration is supported'))
       
   



        
    
if __name__ == '__main__':
    
    
    import matplotlib.pyplot as plt
    from skimage import data
    from scipy.ndimage import fourier_shift
    
    image = data.camera()
    shift0 = np.array([-22.4, 13.32])

    offset_image = fourier_shift(np.fft.fftn(image), shift0)
    
    offset_image = np.fft.ifftn(offset_image).real
    
    shift1 = np.array([15, 5])
    offset_image1 = fourier_shift(np.fft.fftn(offset_image), shift1)
    
    offset_image1 = np.fft.ifftn(offset_image1).real
    
    stack0 = np.array([image,offset_image,offset_image1])
    
    registered1 = stack_registration(stack0, 0 )

    fig = plt.figure(figsize=(8, 3))
    #plt.imshow(offset_image.real, cmap='gray')
    plt.imshow(registered1[1,:,:], cmap='gray', vmin = -24, vmax =270)

