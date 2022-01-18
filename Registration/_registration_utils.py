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


def update_position(old_pos, initial_pos, dx_list, dy_list ):
    
    new_positions = []
    new_lengths = []
    roi_idx = 0
    for pos, pos0, dx, dy in zip(old_pos, initial_pos, dx_list, dy_list, ):
        x1 = pos[0] + dx
        y1 = pos[1] + dy
        x0 = pos0[0]
        y0 = pos0[1]
        dr = np.sqrt(dx**2+dy**2)
        new_positions.append([x1,y1])
        new_lengths.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
        print(f'Displacement for ROI{roi_idx}: {dr:.3f}')    
        roi_idx +=1
        
    return new_positions, new_lengths


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

    
def save_in_excel (filename_xls, sheets_number, **kwargs):
    """
    Creates or overwrite an an excel file with a number of sheets.
    The data to save (in columns) are passed as kwargs as 2D lists or 2D numpy array
    """    
    writer = pd.ExcelWriter(filename_xls)
        
    for sheet_idx in range(sheets_number):  
 
        data = []
        headers = []    
        
        for key, val in kwargs.items():
            val_array = np.array(val)
            data.append(val_array[:,sheet_idx])
            headers.append(key)
            
        df = pd.DataFrame(data, index = headers).transpose()
    
        df.index.name = 't_index'
        df.to_excel(writer, f'ROI_{sheet_idx}')
    writer.save()