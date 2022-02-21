# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:34:41 2022

@author: Andrea Bassi @Polimi
"""
import napari
from qtpy.QtWidgets import QLabel, QVBoxLayout,QSplitter, QHBoxLayout, QWidget, QPushButton
from napari.layers import Image
import numpy as np
from numpy.fft import ifft2, fft2, fftshift 
from napari.qt.threading import thread_worker
from hexSimProcessor import HexSimProcessor
from simProcessor import SimProcessor 
from magicgui import magicgui
from widget_settings import Settings, add_timer
import warnings
import pathlib
from  registration_tools import stack_registration
import pandas as pd

MYPATH ='C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\DatasetTestNapari\\220114_113154_PROCHIP_SIM_ROI.h5'

class HexSIM_MultipleStack():
    
    name = 'HexSIM_Analysis'

    def __init__(self, viewer:napari.Viewer,
                 ):
        self.viewer = viewer
        self.phases_number = 3
        self.pixelsize = 6.5 # TODO make it generic
        
    
        
    def open_table(self,
               filename:pathlib.Path,
               worksheet:str,
               time_tag:str=''):
        
        df = pd.read_excel(filename)
        
        #self.h5path = filename
        self.datasets = df['dataset'].tolist()
        self.alpha = df['alpha'].tolist()
        self.eta = df['eta'].tolist()
        self.beta = df['beta'].tolist()
        self.w = df['w'].tolist()
        self.NA = df['NA'].tolist()
        self.slice = df['slice'].tolist()
        self.M = df['M'].tolist()
        self.n = df['n'].tolist()
        self.wavelength = df['wavelength'].tolist()
        self.filenumber = df['filenumber'].tolist()
        self.date = df['date'].tolist()
        self.time = df['time'].tolist()
        self.type = df['type'].tolist()
        
    def select_h5_file(self, path: pathlib.Path = MYPATH ):    
        self.h5path = path
        
        
        
    def process_all_stacks(self, group:int = 10): 
        self.group = group
        self.start_sim_processor()
        
        for dataset_idx, dataset in enumerate(self.datasets):
            hyperstack = self.open_h5_dataset(self.h5path, dataset)
            alpha = self.alpha[dataset_idx]
            eta = self.eta[dataset_idx]
            beta= self.beta[dataset_idx]
            w= self.w[dataset_idx]
            NA = self.NA[dataset_idx]
            n = self.n[dataset_idx]
            wavelength = self.wavelength[dataset_idx]
            centralslice =self.slice[dataset_idx]
            M = self.M[dataset_idx]
            self.setReconstructor(alpha =alpha,
                                  beta = beta,
                                  eta= eta, w=w, NA= NA, M=M,n=n,
                                  wavelength = wavelength)
            SIMstack = self.stack_reconstruction(hyperstack)
            
   
    def open_h5_dataset(self, path,dataset):
        # open file
        from get_h5_data import get_multiple_h5_datasets, get_h5_attr, get_datasets_index_by_name, get_group_name
        import os
        directory, filename = os.path.split(path)
        t_idx = f'/t{dataset:04d}/'
        index_list, names = get_datasets_index_by_name(path, t_idx)
        stack,found = get_multiple_h5_datasets(path, index_list)
        sp,sz,sy,sx = stack.shape
        if sp != 3 and sp != 7:  
            raise(ValueError(f'Unsupported number of phases. Unable to open dataset {dataset}.'))
        else:
            print(f'\nCorrectly opened dataset {dataset}/{(found//sp)-1} \
                         \nwith {sp} phases and {sz} images')
        
        #updates settings
        measurement_names,_ = get_group_name(path, 'measurement')
        measurement_name = measurement_names[0]
        for key in ['magnification','n','NA','pixelsize','wavelength']:
            val = get_h5_attr(path, key, group = measurement_name) # Checks if the key is in the Scopefoundry measurement settings
            if len(val)>0 and hasattr(self,key):
                new_value = val[0]
                setattr(getattr(self,key), 'val', new_value)
                print(f'Updated {key} to: {new_value} ')
        # imagelayer = Image(stack, name = f'dts{dataset}_{filename}') 
        sp,sz,sy,sx = stack.shape
        assert sy == sx, 'Non-square images are not supported'
        fullname = f'dts{dataset}_{filename}'
        return stack
            
    
    def start_sim_processor(self):     
        self.isCalibrated = False
        
        if hasattr(self, 'h'):
            self.stop_sim_processor()
            self.start_sim_processor()
        else:
            if self.phases_number == 7: 
                self.h = HexSimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor() 
                self.kx_input = np.zeros((3, 1), dtype=np.single)
                self.ky_input = np.zeros((3, 1), dtype=np.single)
                self.p_input = np.zeros((3, 1), dtype=np.single)
                self.ampl_input = np.zeros((3, 1), dtype=np.single)
            elif self.phases_number == 3:
                self.h = SimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor() 
                self.kx_input = np.single(0.0)
                self.ky_input = np.single(0.0)
                self.p_input = np.single(0.0)
                self.ampl_input = np.single(0.0)
            else: 
                raise(ValueError("Invalid number of phases"))
            
            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')
  
    
    def reset_processor(self,*args):
        
        self.isCalibrated = False
        self.stop_sim_processor()
        self.start_sim_processor()
           
        
    def setReconstructor(self, M, NA, alpha, beta, eta,w, n, wavelength):
        
        pixelsize = self.pixelsize
        self.h.usePhases = True
        self.h.debug = False
        self.h.cleanup = False
        self.h.axial = True
        self.h.usemodulation = False
        self.h.magnification = M
        self.h.NA = NA
        self.h.n = n
        self.h.wavelength = wavelength
        self.h.pixelsize = pixelsize
        self.h.alpha = alpha
        self.h.beta = beta
        self.h.w = w
        self.h.eta = eta
            
    def calculate_WF_image(self):
        imageWFdata = np.mean(self.get_imageRaw(), axis=0)
        imname = 'WF_' + self.imageRaw_name
        sz,sy,sx = imageWFdata.shape
        
    
    def calibration(self,data, frame_idx):  
        if hasattr(self, 'h'):
            sp,sz,sy,sx = data.shape
            idx = frame_idx
            delta = self.group // 2
            remainer = self.group % 2
            zmin = max(idx-delta,0)
            zmax = min(idx+delta+remainer,sz)
            new_delta = zmax-zmin
            data = data[:,zmin:zmax,:,:]

            selected_imRaw = np.swapaxes(data, 0, 1).reshape((sp * new_delta, sy, sx))
            self.h.calibrate(selected_imRaw, find_carrier =True)          
            self.isCalibrated = True
            
            
    def stack_reconstruction(self, hyperstack):
               
            sp,sz,sy,sx = hyperstack.shape
            stackSIM = np.zeros([sz,2*sy,2*sx], dtype=np.single)
            
            for zidx in range(sz):
                self.calibration(hyperstack,zidx)
                phases_stack = hyperstack[:,zidx,:,:]
                stackSIM[zidx,:,:] = self.h.reconstruct_rfftw(phases_stack)
            
            self.showCalibrationTable()
            
            return stackSIM
             
    
    def showCalibrationTable(self):
        import pandas as pd
        headers= ['kx_in','ky_in','kx','ky','phase','amplitude']
        vals = [self.kx_input, self.ky_input,
                self.h.kx, self.h.ky,
                self.h.p,self.h.ampl]

        table = pd.DataFrame([vals] , columns = headers )
        print(table)
        
        
    def register_stack(self,image:Image, mode='Euclidean'):
    
        def add_image(data):
            self.viewer.add_image(data, 
                                scale = image.scale,
                                interpolation = 'bilinear',
                                name = f'registered_{image.name}')
            print('Registration completed')
        
        @thread_worker(connect={'returned': add_image})
        def _register_stack():
            frame_idx = self.frame_index.val
            stack = image.data
            registered = stack_registration(stack, z_idx=frame_idx, c_idx=0, method = 'cv2', mode=mode)
            return registered
            
        _register_stack() 

       
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    widget = HexSIM_MultipleStack(viewer)
    mode={"choices": ['Translation','Affine','Euclidean','Homography']}
    registration = magicgui(widget.register_stack, call_button='Register stack', mode=mode)
    xls_opener = magicgui(widget.open_table, call_button='Open excel file', )# call_button='Select image layer')
    h5_opener = magicgui(widget.select_h5_file, call_button='Open h5 dataset')
    process = magicgui(widget.process_all_stacks, call_button='Process stack')
    
    
    viewer.window.add_dock_widget(xls_opener,
                                  name = 'Excel file selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(process,
                                  name = 'Process data',
                                  add_vertical_stretch = True)
    
    
    napari.run()      