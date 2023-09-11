# -*- coding: utf-8 -*-
"""
Created on Fri Jun  22 16:16:06 2023

@author: Andrea Bassi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

def norm(y):
        delta = np.amax(y) - np.amin(y)
        return (y - np.amin(y)) / delta , delta  

class Dataset:
    """
    Class to post process data acquired with napari_roi_registration plugin
    """
    def __init__(self, file, t, intensity, length, group_name, roi):
        self.file = file
        self.t = t
        self.raw = intensity
        self.length = length
        self.roi = roi
        self.group_name = group_name
        self.correct_decay()
        self.normalize() 
    
    def add_field(self, field_name, field):
        setattr(self, field_name, field)
 
    def correct_decay(self, exclude_outliers = True):
        val = self.raw
        t = self.t
        order = 1
        coeff = np.polyfit(t, val, order) 
        fit_function = np.poly1d(coeff)
        val = val - fit_function(t)      
        if exclude_outliers:
            M=3
            amin = np.mean(val) - M*np.std(val)
            amax = np.mean(val) + M*np.std(val)
            val = np.clip(val, amin, amax)
        self.corrected = val# - np.mean(val)
        
    def normalize(self):
        val = self.corrected
        _normalized, _ = norm(val)
        _normalized = _normalized - 0.5
        self.normalized = _normalized


def create_datasets(excel_file,group_name, sampling_time, pixel_size):
    sheets = pd.read_excel(excel_file, sheet_name = None)
    datasets = []
    for roi_name, sheet in sheets.items():
        t_series = sheet['t_index']
        values_series = sheet['intensity']
        length_series = sheet['length']
        dataset = Dataset(excel_file, 
                        np.asarray(t_series.values)*sampling_time, 
                        np.asarray(values_series.values),
                        np.asarray(length_series.values)*pixel_size,
                        group_name, roi_name)
        datasets.append(dataset)
    return datasets


def get_data(groups, folder, sampling_time, pixel_size): 
    all_datasets = []
    for group in groups:   
        filenames = [ x for x in os.listdir(folder) if f'{group}' in x ]
        for filename in filenames:
            path = os.path.join(folder,filename)
            datasets = create_datasets(path, group, sampling_time, pixel_size)
            all_datasets.extend(datasets)
    return all_datasets


def save_in_excel(filename_xlsx, x, y, z, groups, xlabel, ylabel, zlabel):
    writer = pd.ExcelWriter(filename_xlsx)
    for sheet_idx in range(len(groups)):
        
        table = pd.DataFrame(list(zip(x, y[sheet_idx], z[sheet_idx])),
                           #index = x,
                           #headers = text
                           )
        table.columns =[xlabel, ylabel,zlabel]
        sheet_name = groups[sheet_idx]
        
        table.to_excel(writer, f'{sheet_name}_{sheet_idx}')    
    writer.save()




def compute_power_spectrum(datasets, values_x = 't', values_y = 'raw', normalize= True):
    '''returns a new dataset with the powerspectrum of the object'''   
    for dataset in datasets:
        t = getattr(datasets[0],values_x)
        val = getattr(dataset,values_y)
        df = 1/np.amax(t)/2
        size = len(t)
        fftspace = np.linspace(-df*size, +df*size, size)
        fft_val = (fftshift(fft(ifftshift(val))))
        power_spectrum = np.abs(fft_val)**2
        if normalize:
            power_spectrum,_ = norm(power_spectrum)
        dataset.add_field('frequencies', fftspace)
        dataset.add_field('power_spectrum', power_spectrum)


def calculate_means(datasets, groups, values_x = 't', values_y = 'raw' ):
    _values = getattr(datasets[0],values_y)
    x = getattr(datasets[0],values_x)
    sums = [np.zeros(_values.shape)]*len(groups)
    squared = [np.zeros(_values.shape)]*len(groups)
    num = [0]*len(groups)
    for dataset in datasets:
        values = getattr(dataset,values_y)
        group = dataset.group_name
        group_index = groups.index(group)
        sums[group_index] = sums[group_index] + values
        num[group_index] = num[group_index] + 1
    means = [np.zeros(_values.shape)]*len(groups)
    for group_idx, group in enumerate(groups):
        means[group_idx] = sums[group_idx]/num[group_idx]
    num = [0.0]*len(groups)
    for dataset in datasets:
        values = getattr(dataset,values_y)
        group = dataset.group_name
        group_index = groups.index(group)
        squared[group_index] = squared[group_index] + (values-means[group_index])**2
        num[group_index] = num[group_index] + 1
    stds = [np.zeros(_values.shape)]*len(groups)
    for group_idx, group in enumerate(groups):
        stds[group_idx] = np.sqrt(squared[group_idx])/num[group_idx]
    return x, means, stds, num 


def select_by_length(datasets,groups, rmin, rmax):
        new_dataset = []
        num = [0]*len(groups)
        for dataset in datasets:
            max_length = np.amax(dataset.length)
            group = dataset.group_name
            group_index = groups.index(group)
            
            if  max_length>=rmin and max_length <rmax :
                new_dataset.append(dataset)
                num[group_index] = num[group_index] + 1
        for idx,group in enumerate(groups):
            print(f'found {num[idx]} elements in group {group} with length in {rmin}-{rmax}um')
        return new_dataset

def plot_datasets(datasets, title, xlabel, ylabel, values_x = 't', values_y = 'raw' ):
     
     x = getattr(datasets[0],values_x)
     y_list = []
     for dataset in datasets:
         y = getattr(dataset,values_y)
         y_list.append(y)
     plot_data(x, y_list, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_data(xvalues, yvalues, yerr=None, title = 'mytitle', xlabel = 'x', ylabel = 'y', figtype = 'linear', 
                normalize = False, legend = None,
                xmin = None, xmax = None,
                vmin = None, vmax = None):
    char_size = 10
    linewidth = 0.65
    plt.rc('font', family='calibri', size=char_size)
    fig = plt.figure(figsize=(4,3), dpi=300)
    ax = fig.add_subplot(111)
    COLORS = ['grey','darkslategray','black',]
    plot_count = 0
    for val_idx, val in enumerate(yvalues):
        
        deltaA = 1    
        if normalize:
            #val, deltaA = norm(val)
            deltaA = np.amax(val)-np.amin(val) 
            val = val/np.amax(val)
            #val = val/np.mean(val)
            #val = val - np.mean(val)
        if yerr != None:
            err = yerr[val_idx]/deltaA # normalized by delta amplitude, if normalized is True 
            #err = yerr[val_idx]
            ax.errorbar(xvalues, val, err,
                        linewidth=linewidth,
                        elinewidth=0.15*linewidth, 
                        capsize=1.2*linewidth,
                        color = COLORS[plot_count%len(COLORS)])
            if xmin !=None and xmax!=None:
                ax.set_xlim([xmin, xmax])
            if vmin !=None and vmax!=None:
                ax.set_ylim([vmin, vmax])
        else:    
            ax.plot(xvalues, val, 
            linewidth=linewidth)
        plot_count += 1
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
    if legend !=None:
        ax.legend(legend,
                    loc='best', frameon = False,
                    fontsize=char_size*0.8)   
    fig.tight_layout()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()

if __name__== "__main__":

    from magicgui import magicgui
    import pathlib
    import os

    FOLDER = os.path.join(os.path.dirname(__file__),'Data')

    @magicgui(
        call_button="Process",
        preprocessing={"choices": ['raw', 'corrected', 'normalized']},
        folder={'mode': 'd'},
        saving_folder={'mode': 'd'},
    )
    def processing_ui(
        folder=pathlib.Path(FOLDER),
        groups='Col-0,aca2-2',
        preprocessing='normalized',
        min_length: float =0.0,
        max_length: float =100.0,
        show_preprocessed: bool = False,
        show_lengths: bool = True,
        show_spectrum:bool = True,
        normalize_spectra: bool = False,
        save_data:bool = False, 
        saving_folder=pathlib.Path(FOLDER), 
        ):

        groups_list = groups.split(",")

        original_datasets = get_data(groups_list, folder=folder,
                                 sampling_time=3.0, pixel_size=0.65)
        
        mydatasets = select_by_length(original_datasets,groups_list, min_length, max_length)

        if show_preprocessed:
            plot_datasets(mydatasets, title=preprocessing,
                        xlabel='time(s)',
                        ylabel='intensity(a.u)', values_x = 't', values_y = preprocessing)

        if show_lengths:
            t, mean_length, std_length, _num = calculate_means(mydatasets, groups_list,
                                            values_x = 't',
                                            values_y = 'length')
            plot_data(t, mean_length, yerr=std_length,
                    title = 'Lengths by group',
                    xlabel='time (s)', ylabel = 'length (\u03BCm)',
                    normalize = False, legend = groups_list)
        
        
        compute_power_spectrum(mydatasets,values_y = preprocessing, normalize=normalize_spectra)
        freqs, mean_spectrum, std_spectrum, _num = calculate_means(mydatasets, groups_list,
                                        values_x = 'frequencies',
                                        values_y = 'power_spectrum')
        if show_spectrum:
            plot_data(freqs, mean_spectrum, yerr=std_spectrum, figtype = 'log',
                    title = 'Fourier Spectrum',
                    xlabel='frequency (Hz)', 
                    ylabel = 'power spectrum',
                    normalize = False, legend = groups_list,
                    xmin = 0.001,
                    xmax = 0.151)
        if save_data:
            save_filename = os.path.join(saving_folder,
                                         '_vs_'.join(map(str,groups_list))+'.xlsx',
                                         
                                         )
            save_in_excel(save_filename, 
                          x = freqs, y = mean_spectrum, z = std_spectrum,
                          groups = groups_list,
                          xlabel = 'freqs',
                          ylabel = 'spectrum',
                          zlabel = 'error')
        
    processing_ui.show(run=True)
    




    
