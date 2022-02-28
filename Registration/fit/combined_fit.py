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


def concatenate(arrays_list):
    conc = np.concatenate(arrays_list)
    return conc.ravel()
    


def Diffusion(times, t0, r0, r1, p0, p1, v0, v1, A0, A1):

    #t0 = np.array((t0,t1)) 
    t0=30
    
    v_list = (v0,v1)
    A_list = (A0,A1)
    #print (A_list)
    p_list = (p0,p1)
    #c = np.array((c0,c1))
    r_list = (r0,r1)
    #t0_list = (t00,t01)
    #i = A/np.sqrt(t-t0)*np.exp(-p/(t-t0)) + c
    #x = np.array( (,) ) 
    # i = np.exp(v/2*(k-v/2*t)) * A /np.sqrt(t)*np.exp(-p/t) 
    i_list = []
    starting_time = 0
    for roi_idx in range(len(samples_num)):
        t0 = 32 # t00 #= t0_list[roi_idx]
        delta_t = samples_num[roi_idx]
        t = times[starting_time:starting_time+delta_t] - t0
        A = A_list[roi_idx]
        r = r_list[roi_idx]
        v = v_list[roi_idx]
        p = p_list[roi_idx]
        with np.errstate(over='ignore', invalid='ignore'):
            i = np.exp(p*v/2*(r-v/2*t)) * A * np.sqrt(p/(4*np.pi*t))*np.exp(-p*r**2/(4*t))
        i_list.append(i)
        starting_time = delta_t
    
    return concatenate(i_list)

if __name__ == '__main__':
    
    guess = [30, 50, 45, 1.5, 1.5, 50, 4, 854, 882]
    #guess = [0.5, 0.4, 0.1, 1, 0.1, 0.2, 0, 0]
    
    excel_file = os.getcwd() +'\\test10.xlsx'
    t_indices_array, yx_array, intensities_array = take_excel_data(excel_file)
    
    yx_stimolo = np.array([347,687])
    yx_roi0 = yx_array[0,0,:]
    yx_roi1 = yx_array[1,0,:]
    
    d0 = np.sqrt(np.sum((yx_roi0-yx_stimolo)**2)) 
    d1 = np.sqrt(np.sum((yx_roi1-yx_stimolo)**2)) 
    
    d= (d0,d1)
    
    rois_num = intensities_array.shape[1]
    intensities_array = intensities_array - intensities_array[0,:]
    
    max_indices = np.argmax(intensities_array, axis= 0)
    
    left_thresold = 0.5
    right_thresold = 0.2
    left_idx =  np.zeros(rois_num, dtype = int)
    right_idx= np.zeros(rois_num, dtype = int)
    
    times = [] #np.zeros([t_max-t_min, rois_num],dtype=int)
    intensities = [] # np.zeros([t_max-t_min, rois_num])
    samples_num = []
        
    for roi_idx in range(rois_num): 
        iarray = intensities_array[:,roi_idx]
        max_intensity = np.amax(iarray)
        max_idx = max_indices[roi_idx]
        left_val = max_intensity*left_thresold
        right_val = max_intensity*right_thresold
        
        t_min = 1
        for ti in range(max_idx,-1,-1):
            #print(ti)
            if iarray[ti] > left_val:
                #print(iarray[ti])
                t_min-=1
            else:
                break
        t_max = 0
        for ti in range(max_idx, iarray.size):
            if iarray[ti] > right_val:
                #print(iarray[ti])
                t_max+=1
            else:
                break
        
        t_min = -2
        t_max = 7
    
        tis = t_indices_array[max_idx+t_min:max_idx+t_max, roi_idx]
        times.append(tis)
        
        samples_num.append(len(tis))
        intensities.append(intensities_array[tis,roi_idx])
           
    # parameters, covariance = curve_fit(Diffusion, concatenate(times),
    #                                      concatenate(intensities),
    #                                      p0 = guess)
    
    parameters = guess
    print(*parameters)
    
    fig = plt.figure(figsize=(4,3), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(t_indices_array, intensities_array,
             '-o', label='data',
             markersize=2,
             linewidth=0.3)
    fitted_intensities = Diffusion(concatenate(times), *parameters)
    previuos_tmax = 0
    for roi_idx in range(rois_num): 
        #fitted_intensities = np.reshape(fitted_intensities, intensities.shape)
        
        delta_t = samples_num[roi_idx]
        ts = concatenate(times)[previuos_tmax:previuos_tmax+delta_t]
        fi = fitted_intensities[previuos_tmax:previuos_tmax+delta_t]
        #fi = concatenate(intensities)[previuos_tmax:previuos_tmax+delta_t]
        
        plt.plot(ts, fi, linestyle='dashed', label='fit')
        previuos_tmax = len(ts)
    
    ax.set_xlabel('t_index')
    ax.set_ylabel('intensity')
    ax.grid()
    
    #ax.set_yscale('log'), ax.set_ylim(1)
    ax.legend() 