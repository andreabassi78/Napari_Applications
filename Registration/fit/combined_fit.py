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

def Diffusion(t_and_grp, p, t0, A0, A1, c0, c1):
    
    t = t_and_grp[:,0]
    grp_id = t_and_grp[:,1]
    A = np.array([[A0,A1][int(gid)] for gid in grp_id])
    c = np.array([[c0,c1][int(gid)] for gid in grp_id])
    
    i = A/np.sqrt(t-t0)*np.exp(-p/(t-t0)) + c
    return i

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
    
    pass
    
    intensities_lists = []
    t_indices_lists = []
    
    t_max = 7
    t_min = -3
    
    excel_file = os.getcwd() +'\\test7.xlsx'
    t_indices_array, yx_array, intensities_array = take_excel_data(excel_file)
    
    guess = [3, 33, 3500, 1300, -700, -300]
    
    max_indices = np.argmax(intensities_array, axis= 0)
    
    rois_num = intensities_array.shape[1]
    
    
    for roi_idx in range (rois_num):
        
        intensities_array[...,roi_idx] = intensities_array[...,roi_idx] - (intensities_array[0, roi_idx])  
        intensities_data = intensities_array[max_indices[roi_idx]+t_min:max_indices[roi_idx]+t_max, roi_idx]
        t_indices_data = t_indices_array[max_indices[roi_idx]+t_min:max_indices[roi_idx]+t_max, roi_idx]
        if roi_idx == 0:
            intensities= intensities_data
            t_indices = t_indices_data
        else:
            intensities_all = np.concatenate((intensities,intensities_data))
            t_indices_all = np.concatenate((t_indices,t_indices_data))
        
    
    t_and_grp_all = np.zeros((t_indices_all.size, 2))
    t_and_grp_all[:,0] = t_indices_all 
    t_and_grp_all[0:10, 1] = 0
    t_and_grp_all[10:20, 1] = 1
    
    parameters, covariance = curve_fit(Diffusion, t_and_grp_all, intensities_all, p0 = guess)
    #print('parameters:', parameters)
    
    #for gid,color, roi_idx in zip([0,1],['r','k'], range(rois_num)):
    plt.plot(t_indices_array, intensities_array, 'o', label='data')
    # A = parameters[3+gid]
    # t_and_grp = np.column_stack([t_and_grp_all, np.ones_like(t_and_grp_all)*gid])
    plt.plot(t_and_grp_all[:,0], Diffusion(t_and_grp_all, *parameters),
              linestyle='dashed', label='fit')

    plt.xlabel('t_index')
    plt.ylabel('intensity')
    plt.legend()