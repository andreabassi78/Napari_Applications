# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:47:58 2022

@author: GiorgiaT
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Diffusion(times, p, t00, t01,  A0, A1, c0, c1):
    
    t = times
    t0 = np.array((t00,t01))
    A = np.array((A0,A1))
    # p = np.array((p0,p1)) 
    c = np.array((c0,c1)) 
    i = A/np.sqrt(t-t0)*np.exp(-p/(t-t0)) + c
    
    return i.ravel()

def take_excel_data(excel_file):

    file = pd.read_excel(excel_file, sheet_name = None)
 
    intensities = []
    yx_coordinates = []
    t_indices = []

    for name, sheet in file.items():
        t_index = sheet[['t_index']].values
        intensity = sheet[['intensity']].values
        yx = sheet[['y', 'x']].values
        t_indices.append(t_index)
        intensities.append(intensity)
        yx_coordinates.append(yx)
    
    t_indices_array = np.squeeze(np.array(t_indices)).T
    intensities_array = np.squeeze(np.array(intensities)).T
    yx_array = np.array(yx_coordinates)
    
    return t_indices_array, yx_array, intensities_array

if __name__ == '__main__':
     
    t_max = 7
    t_min = -3
    
    guess = [2.4, 37, 40, 2500, 1700, -500, -300]
    
    excel_file = os.getcwd() +'\\test7.xlsx'
    
    t_indices_array, yx_array, intensities_array = take_excel_data(excel_file)
    
    max_indices = np.argmax(intensities_array, axis= 0)
    rois_num = intensities_array.shape[1]
    intensities_array = intensities_array - intensities_array[0,:]
    
    times = np.zeros([t_max-t_min, rois_num],dtype=int)
    intensities = np.zeros([t_max-t_min, rois_num])
    
    for roi_idx in range(rois_num):    
        
        times[:,roi_idx] = t_indices_array[max_indices[roi_idx]+t_min:max_indices[roi_idx]+t_max, roi_idx]
        intensities[:,roi_idx] = intensities_array[times[:,roi_idx],roi_idx]
          
    parameters, covariance = curve_fit(Diffusion, times, intensities.ravel(), p0 = guess)
    print('parameters:', parameters)
    
    plt.plot(t_indices_array, intensities_array, 'o', label='data')
    
    fitted_intensities = Diffusion(times, *parameters)
    fitted_intensities = np.reshape(fitted_intensities, intensities.shape)
    plt.plot(times, fitted_intensities,
              linestyle='dashed', label='fit')

    plt.xlabel('t_index')
    plt.ylabel('intensity')
    plt.legend()