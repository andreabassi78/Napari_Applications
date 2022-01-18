# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:35:46 2021

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normalize_stack(stack, **kwargs):
    '''
    -normalizes n-dimensional stack it to its maximum and minimum values,
    unless normalization values are provided in kwargs,
    -casts the image to 8 bit for fast processing with cv2
    '''
    img = np.float32(stack)
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    else:    
        vmin = np.amin(img)
   
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    else:    
        vmax = np.amax(img)
    saturation = 1   
    img = saturation * (img-vmin) / (vmax-vmin)
    img = (img*255).astype('uint8') 
    return img, vmin, vmax


def filter_image(img, sigma):
    if sigma >0:
        sigma = (sigma//2)*2+1 # sigma must be odd in cv2
        #filtered = cv2.GaussianBlur(img,(sigma,sigma),cv2.BORDER_DEFAULT)
        filtered = cv2.medianBlur(img,sigma)
        return filtered
    else:
        return img
 

def select_rois_from_stack(input_stack, positions, roi_size):
    
    rois = []
    half_size = roi_size//2
    for pos in positions:
        t = int(pos[0]) 
        y = int(pos[1])
        x = int(pos[2])
        rois.append(input_stack[t, y-half_size:y+half_size,
                                   x-half_size:x+half_size])
    return rois
       
    
def select_rois(input_image, positions, roi_size):
    
    rois = []
    half_size = roi_size//2
    for pos in positions:
        x = int(pos[0])
        y = int(pos[1])
        rois.append(input_image[y-half_size:y+half_size,
                                x-half_size:x+half_size])
    return rois
        
    
def align_with_registration(next_rois, previous_rois, filter_size, roi_size):  
    
    original_rois = []
    aligned_rois = []
    dx_list = []
    dy_list = []
    
    half_size = roi_size//2
    
    warp_mode = cv2.MOTION_TRANSLATION 
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations,  termination_eps)
    
    for previous_roi, next_roi in zip(previous_rois, next_rois):
      
        previous_roi = filter_image(previous_roi, filter_size)
        next_roi = filter_image(next_roi, filter_size)
        
        sx,sy = previous_roi.shape
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        try:
            _, warp_matrix = cv2.findTransformECC (previous_roi, next_roi,
                                                      warp_matrix, warp_mode, criteria)
            
            next_roi_aligned = cv2.warpAffine(next_roi, warp_matrix, (sx,sy),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except:
            next_roi_aligned = next_roi
        
        original_roi = previous_roi[sy//2-half_size:sy//2+half_size,
                                        sx//2-half_size:sx//2+half_size ]
        
        aligned_roi =  next_roi_aligned[sy//2-half_size:sy//2+half_size,
                                        sx//2-half_size:sx//2+half_size ]
    
        original_rois.append(original_roi)
        aligned_rois.append(aligned_roi)
        
        dx = warp_matrix[0,2]
        dy = warp_matrix[1,2]
        
        dx_list.append(dx)
        dy_list.append(dy)
    
    return aligned_rois, original_rois, dx_list, dy_list


def update_position(pos_list, dz, dx_list, dy_list ):
    
    next_pos_list = []
    for pos, dx, dy in zip(pos_list, dx_list, dy_list):
        z1 = pos[0] + dz
        y1 = pos[1] + dy
        x1 = pos[2] + dx
        next_pos_list.append([z1,y1,x1])  
        
    return next_pos_list


def rectangle(center, sidey, sidex):
    cz=center[0]
    cy=center[1]
    cx=center[2]
    hsx = sidex//2
    hsy = sidey//2
    rect = [ [cz, cy+hsy, cx-hsx], # up-left
             [cz, cy+hsy, cx+hsx], # up-right
             [cz, cy-hsy, cx+hsx], # down-right
             [cz, cy-hsy, cx-hsx]  # down-left
           ] 
    return rect


def correct_decay(data):
    '''
    corrects decay fitting data with a polynomial and subtracting it
    data are organized as a list (time) of list (roi)
    returns a 2D numpy array
    '''
    data = np.array(data) # shape is raws:time, cols:roi
    rows, cols = data.shape 
    order = 2
    corrected = np.zeros_like(data)
    for col_idx, column in enumerate(data.transpose()):
        t_data = range(rows)
        coeff = np.polyfit(t_data, column, order) 
        fit_function = np.poly1d(coeff)
        corrected_value = column - fit_function(t_data) 
        corrected[:,col_idx]= corrected_value
    
    return corrected


def calculate_spectrum(data):
    '''
    calculates power spectrum with fft
    data are organized as a list (time) of list (roi), or as a 2D numpy array
    returns a 2D numpy array
    '''
    data = np.array(data) # shape is raws:time, cols:roi
    ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data), axis=0))
    spectra = (np.abs(ft))**2 
    
    return spectra
    

def plot_data(data, xlabel, ylabel, plot_type='lin'):
    '''
    data are organized as a list (time) of list (roi), or as a 2D numpy array
    '''
    data = np.array(data)
    roi_num = data.shape[1]
    legend = [f'ROI {roi_idx}' for roi_idx in range(roi_num)]
    char_size = 10
    linewidth = 0.85
    plt.rc('font', family='calibri', size=char_size)
    fig = plt.figure(figsize=(4,3), dpi=150)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_title(title, size=char_size)   
    ax.set_xlabel(xlabel, size=char_size)
    ax.set_ylabel(ylabel, size=char_size)
    if plot_type == 'log':
        # data=data+np.mean(data) # in case of log plot, the mean is added to avoid zeros
        ax.set_yscale('log')
    ax.plot(data, linewidth=linewidth)
    ax.xaxis.set_tick_params(labelsize=char_size*0.75)
    ax.yaxis.set_tick_params(labelsize=char_size*0.75)
    ax.legend(legend, loc='best', frameon = False, fontsize=char_size*0.8)
    ax.grid(True, which='major',axis='both',alpha=0.2)
    fig.tight_layout()
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)


def save_in_excel(filename_xls, sheet_name, **kwargs):

    headers = list(kwargs.keys())
    values = np.array(list(kwargs.values())) # consider using np.fromiter
    
    writer = pd.ExcelWriter(filename_xls)
    
    for sheet_idx in range(values.shape[2]):
        table = pd.DataFrame(values[...,sheet_idx],
                          index = headers
                          ).transpose()
        table.index.name = 't_index'
        table.to_excel(writer, f'{sheet_name}_{sheet_idx}')
        # print(table)
        
    writer.save()