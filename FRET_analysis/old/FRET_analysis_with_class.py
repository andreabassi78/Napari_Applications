# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:35:43 2021

@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import pandas as pd
import matplotlib.pyplot as plt

def norm(y):
    delta = np.amax(y) - np.amin(y)
    return (y - np.amin(y)) / delta , delta  
      
class dataset:
    
    def __init__(self, data_type = 'test', axis_type = 'time', axis_unit = 's'):
        """
        Structures the dataset so to have an axis (abscissa in the plot), 
        with the specified unit and type
        The values are stored in a dictionary with 2 keys, by the other functions       
        """
        self.data_type = data_type
        self.axis_type = axis_type # horizontal label in the figures
        self.axis_unit = axis_unit        
        self.values_unit = 'au'
        self.values = {}        
        self.data_count = 0
        self.position = ''
        self.length = 0
        self.mutant = '' 
        
    def get_element_values_by_key(self, label0, label1):
        """
        get the values of a dataset 
        corresponging to a certain key (label0) and subkey (label1).
        If multiple keys and subkeys are present, it returns the first encountered
        """
        for key, val in self.values.items():
            if label0 in key:
                for subkey, subval in val.items():
                    if label1 in subkey:
                        return(subval)
        return None            
                    
    def get_element_by_key(self, label0, label1):
        """
        get a new dataset containing a single element 
        corresponging to a certain key (label0) and subkey (label1).
        If multiple keys and subkeys are present, it returns the first encountered
        
        """
        for key, val in self.values.items():
            if label0 in key:
                for subkey, subval in val.items():
                    if label1 in subkey:
                        element = dataset(self.data_type, self.axis_type, self.axis_unit)
                        element.values[key] = {subkey:subval}
                        element.axis = self.axis
                        element.values_unit = self.values_unit
                        return element            
    
    def get_elements_list(self):
        ''' 
        returns a list for the keys and subkey (in a 2 element-list) 
        of the found elements
        '''
        element_list = []
        for key, val in self.values.items():
            for subkey, subval in val.items():
                    element_list.append([key,subkey])
        return element_list
    
    def select_by_length(self, dr, rmin, rmax):
        
        data_new = dataset(f'{self.data_type}', self.axis_type, self.axis_unit )
        data_new.axis = self.axis 
        data_new.values_unit = self.values_unit
        dr_new = dataset(f'{dr.data_type}', dr.axis_type, dr.axis_unit )
        dr_new.axis = dr.axis 
        dr_new.values_unit = dr.values_unit
        
        for key, val in self.values.items():
            data_new.values[key] = {} 
            dr_new.values[key] = {}  
            for subkey, subval in val.items():
                length = np.amax(dr.values[key][subkey])
                if  length>=rmin and length <rmax :
                    data_new.values[key][subkey] = subval
                    dr_new.values[key][subkey] = dr.values[key][subkey]
        return data_new, dr_new
                      
            
    
    @classmethod
    def get_values_from_file(cls,path, 
                             mutants, positions, exclude_position, 
                             size, deltaT, normalize = False):
        '''
        opens data stored in sheets (different samples) 
        and columns (different radical hairs in the same sample)
        '''
        xls = pd.ExcelFile(path)
        sheets = pd.read_excel(xls, sheet_name = None )
        
        FRETdata = cls('FRET ratio','time', 's')
        
        FRETdata.axis = np.linspace(0, deltaT*(size-1), size)  
        
        dr = cls('\u0394L', 'time', 's' )
        dr.values_unit = '\u03BCm' 
        dr.axis = FRETdata.axis
        
        for key, sheet in sheets.items():
        
            for mutant in mutants:
                
                if mutant in key: 
                    
                    # if not key in data.keys():
                    FRETdata.values[key] = {} 
                    dr.values[key] = {}
                        
                    columns = sheet.columns
                    for cidx,column in enumerate(columns):
                        
                        for position in positions:
                            
                            if position in column:
                                
                                if not exclude_position in column:
                                
                                    val = sheet[column]
                                    dx = sheet[columns[cidx+1]]
                                    if len(columns)> cidx+2:
                                        dy = sheet[columns[cidx+2]]
                                    else:
                                        dy = np.zeros(size+1)
                                    val = val[0:size]
                                    dx = dx[1:size+1]
                                    dy = dy[1:size+1]
                                    
                                    dx = dx.fillna(0.0).to_numpy()
                                    if len(columns)> cidx+2:
                                        dy = dy.fillna(0.0).to_numpy()
                                    
                                    drho = np.sqrt(dx**2+dy**2) 
                                    
                                    
                                    if normalize:
                                        val, _ = norm(val)
                                        val = val - 0.5
                                                                       
                                    FRETdata.values[key][column] = val.to_numpy()
                                    dr.values[key][column] = drho
                                    # print(key, column)
                                    FRETdata.data_count += 1
                                    dr.data_count +=1
        return FRETdata, dr                            
         
                                
    def correct_decay(self, normalize = False):
        '''
        fits data with a polynomial and subtracts it
        '''
        order = 1
        for key, val in self.values.items():
            for subkey, subval in val.items():
                t_data = self.axis
                coeff = np.polyfit(t_data, subval, order) 
                fit_function = np.poly1d(coeff)
                corrected = subval - fit_function(t_data) 
                if normalize:
                    corrected, _ = norm(corrected)
                    corrected = corrected - 0.5
                self.values[key][subkey] = corrected   

        
    def compute_fft(self, normalize = False):
        '''returns a new dataset with the powerspectrum of the object'''
        fftdata = dataset('Power spectrum', 'frequency', 'Hz' )
        t = self.axis
        df = 1/np.amax(t)/2
        size = len(t)
        # f = fftfreq(SIZE, DeltaT)
        fftdata.axis = np.linspace(-df*size, +df*size, size)
        
        for key, val in self.values.items():
            
            fftdata.values[key] = {}
        
            for sub_key, sub_val in val.items():
                
                fft_val = (fftshift(fft(ifftshift(sub_val))))
                power_spectrum = np.abs(fft_val)**2
                 
                fftdata.values[key][sub_key] = power_spectrum 
                fftdata.data_count += 1
        
        return fftdata 


    def calculate_mean(self, types, positions):
        '''returns a new dataset with the mean of the object,
        calulated on the selected types and positions'''
        average = dataset(f'{self.data_type} - mean', self.axis_type, self.axis_unit )
        average.axis = self.axis 
        average.values_unit = self.values_unit
        size = len(average.axis)
        values_list = {}
        
        for key, val in self.values.items():
            
            for itype in types:                    
                if itype in key:                        
                    if not itype in average.values.keys():
                        average.values[itype] = {}
                        values_list[itype] = []
                    for sub_key,sub_val in val.items():                            
                        for position in positions:
                            if position in sub_key: 
                                values_list[itype].append(self.values[key][sub_key])
            
        for group in average.values.keys():
            summed = np.zeros(size)
            variance = np.zeros(size)
            for idx, element in enumerate(values_list[group]):
                summed += element
            samples_per_group = len(values_list[group])
            average.values[group]['- mean'] = summed /samples_per_group    
            
            for idx, element in enumerate(values_list[group]):
                variance += element**2 - average.values[group]['- mean']**2
                
            average.values[group]['- sterr'] = np.sqrt(variance)/(samples_per_group)
            
        return average
        
    def plot_data(self, figtype = 'linear', 
                  normalize = False, legend = False,
                  xmin = 0, xmax = 0.1,
                  vmin=0.01, vmax = 1):

        char_size = 10
        linewidth = 0.85
        
        plt.rc('font', family='calibri', size=char_size)
        
        fig = plt.figure(figsize=(4,3), dpi=300)
        ax = fig.add_subplot(111)
    
        title = self.data_type
        xlabel = f'{self.axis_type} ({self.axis_unit})'
        ylabel = f'{self.data_type} ({self.values_unit})'
        
        COLORS = ['grey','darkslategray','black',]
        plot_count = 0
        key_list = []
        
        for key, val in self.values.items():
        
            for sub_key, sub_val in val.items():
                
                axis = self.axis
                
                if sub_key == '- sterr':
                    continue
                
                pos_idx = np.ones(len(sub_val), dtype=bool)
                                
                deltaA = 1    
                if normalize:
                    sub_val, deltaA = norm(sub_val)
                    #sub_val = sub_val - np.mean(sub_val)
                    
                if '- sterr' in val.keys():
                    sterr = val['- sterr'] /deltaA # normalized by delta amplitude, if requested 
                    ax.errorbar(axis[pos_idx], sub_val[pos_idx], sterr[pos_idx],
                                linewidth=linewidth,
                                elinewidth=0.15*linewidth, 
                                capsize=1*linewidth,
                                color = COLORS[plot_count%len(COLORS)])
                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([vmin, vmax])
                else:    
                    ax.plot(axis[pos_idx], sub_val[pos_idx], 
                            linewidth=linewidth)
                
                key_list.append(f'{key} {sub_key}')
                plot_count += 1
                
        print(f'Plotted {self.data_type}:', plot_count)           
        
        #ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, size=char_size)   
        ax.set_xlabel(xlabel, size=char_size)
        ax.set_ylabel(ylabel, size=char_size)
        ax.xaxis.set_tick_params(labelsize=char_size*0.75)
        ax.yaxis.set_tick_params(labelsize=char_size*0.75)
        
        if figtype == 'log':
            ax.set_yscale('log')
            # ax.set_xscale('log')
        
        ax.grid(True, which='major',axis='both',alpha=0.2)   
        
        if legend:
            ax.legend(key_list,
                      loc='best', frameon = False,
                      fontsize=char_size*0.8)
            
        fig.tight_layout()
        plt.rcParams.update(plt.rcParamsDefault)
        plt.show()
     
if __name__ == '__main__': 
       
    file_path = 'SPIM analysis ACA2.2-WT new.xlsx'
    directory = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Documenti\\PythonProjects\\FRET_analysis\\'
    path = directory+file_path
    MUTANTS = ['WT','aca2.2']
    MUTANTS = ['aca']
    POSITIONS = ['Lower', 'Middle', 'Higher'] 
    EXCLUDE_POSITION = 'inverse'
    POSITIONS = ['Lower'] 
    MIN_LENGHT = 0
    MAX_LENGHT = 100
    
    SIZE = 398    
    DeltaT = 3 #s
    
    
    FRETdata, delta = dataset.get_values_from_file(path, MUTANTS, POSITIONS, EXCLUDE_POSITION, 
                                size=SIZE, deltaT=DeltaT, normalize=False)
    
    print('Dataset found:', len(FRETdata.get_elements_list()))
    
    FRETdata,delta = FRETdata.select_by_length(delta, MIN_LENGHT, MAX_LENGHT) 
    
    
    FRETdata.correct_decay(normalize=True)
    spectra = FRETdata.compute_fft()
    
    delta.plot_data(legend=False)
    
    print('Dataset found:', len(delta.get_elements_list()))
    
    FRETdata.plot_data(legend=False)
    spectra.plot_data(figtype = 'log', legend=False)
    
    mean_spectra = spectra.calculate_mean(MUTANTS, POSITIONS)                        
    mean_spectra.plot_data(figtype='log', normalize=False,
                           legend=True,
                           xmin=0, xmax=0.15,
                           vmin=1, vmax=100)               

    # mean_delta = delta.calculate_mean(MUTANTS, POSITIONS)                        
    # mean_delta.plot_data(figtype='lin', normalize=False,
    #                        legend=True,
    #                        xmin=0, xmax=1200,
    #                        vmin=0, vmax=60)               



    # index = 3
    # element_lst = FRETdata.get_elements_list()
    # data_i = FRETdata.get_element_by_key(*element_lst[index])
    # data_i.plot_data(legend = True)
    # fftdata_i = spectra.get_element_by_key(*element_lst[index])
    # fftdata_i.plot_data(figtype = 'log', normalize = False,
    #                     legend = True, cut_negative= True)
    
    
    # index = 1
    # element_lst = delta.get_elements_list()
    # delta_i = delta.get_element_by_key(*element_lst[index])
    # delta_i.plot_data(legend = True)
    
    