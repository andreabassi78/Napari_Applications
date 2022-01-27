# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:34:41 2022

@author: Andrea Bassi @Polimi
"""
import napari
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QTableWidget, QSplitter, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QComboBox, QWidget, QFrame, QLabel, QFormLayout, QVBoxLayout, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox
from skimage.measure import regionprops
from napari.layers import Image, Points, Labels, Shapes
from napari.qt.threading import thread_worker
import json
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import tifffile as tif
from numpy.fft import ifft2, fft2, fftshift
# from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
from qtpy.QtWidgets import QFileDialog, QTableWidgetItem   
# from ScopeFoundry import Measurement
# from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file

from hexSimProcessor import HexSimProcessor
from simProcessor import SimProcessor 
from image_decorr import ImageDecorr
from magicgui import magicgui
from widget_settings import Settings, add_timer
import warnings

class HexSimAnalysis(QWidget):
    
    name = 'HexSIM_Analysis'

    def __init__(self, viewer:napari.Viewer,
                 ):
        self.viewer = viewer
        super().__init__()
        
        self.setup_ui() # run setup_ui before instanciating the Settings
        self.start_sim_processor()
        viewer.dims.events.current_step.connect(self.select_index)
        
    def setup_ui(self):     
        
        def add_section(_layout,_title):
            splitter = QSplitter(Qt.Vertical)
            _layout.addWidget(splitter)
            _layout.addWidget(QLabel(_title))
            
        # initialize layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Settings
        settings_layout = QVBoxLayout()
        add_section(settings_layout,'Settings')
        layout.addLayout(settings_layout)
        self.create_Settings(settings_layout)
        
        # Buttons
        buttons_layout = QVBoxLayout()
        add_section(buttons_layout,'Operations')
        layout.addLayout(buttons_layout)
        self.create_Buttons(buttons_layout)
             
        
    def create_Settings(self, slayout): 
        
        self.phases_number = Settings('phases', dtype=int, initial=3, layout=slayout, 
                              write_function = self.reset_processor)
        self.debug = Settings('debug', dtype=bool, initial=False, layout=slayout,
                          write_function = self.setReconstructor) 
        self.cleanup = Settings('cleanup', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.gpu = Settings('gpu', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.compact = Settings('compact', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.axial = Settings('axial', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.usemodulation = Settings('usemodulation', dtype=bool, initial=True,
                                      layout=slayout, write_function = self.setReconstructor)
        self.magnification = Settings('M', dtype=float, initial=63, unit = 'X',  
                                      layout=slayout, write_function = self.setReconstructor)
        self.NA = Settings('NA', dtype=float, initial=1.00, layout=slayout, 
                                       write_function = self.setReconstructor)
        self.n = Settings(name ='n', dtype=float, initial=1.33,  spinbox_decimals=2,
                                      layout=slayout, write_function = self.setReconstructor)
        self.wavelength = Settings('\u03BB', dtype=float, initial=0.570,
                                       layout=slayout,  spinbox_decimals=3,
                                       write_function = self.setReconstructor)
        self.pixelsize = Settings('pixelsize', dtype=float, initial=6.50, layout=slayout,
                                  spinbox_decimals=2, unit = 'um',
                                  write_function = self.setReconstructor)
        self.alpha = Settings('alpha', dtype=float, initial=0.500,  spinbox_decimals=3, 
                              layout=slayout, write_function = self.setReconstructor)
        self.beta = Settings('beta', dtype=float, initial=0.950, 
                             layout=slayout,  spinbox_decimals=3,
                             write_function = self.setReconstructor)
        self.w = Settings('w', dtype=float, initial=2.00, layout=slayout,
                              spinbox_decimals=2,
                              write_function = self.setReconstructor)
        self.eta = Settings('eta', dtype=float, initial=0.4,
                            layout=slayout, spinbox_decimals=2, spinbox_step=0.01,
                            write_function = self.setReconstructor)
        
        self.use_phases = Settings('use_phases', dtype=bool, initial=True, layout=slayout,                         
                                   write_function = self.setReconstructor)
        self.find_carrier = Settings('Find Carrier', dtype=bool, initial=True,
                                     layout=slayout, 
                                     write_function = self.setReconstructor)
        # self.selectROI = Settings('selectROI', dtype=bool,
        #                           initial=False, layout=slayout) 
        # self.roiX = Settings('roiX', dtype=int,
        #                      initial=600,  layout=slayout)
        # self.roiY = Settings('roiY', dtype=int,
        #                      initial=1200,  layout=slayout)
        # self.ROI_size = Settings('ROI_size', dtype=int, initial=512,
        #                          vmin=1, vmax=2048, layout=slayout) 
        # self.dataset_index = Settings('dataset_index', dtype = int, initial=0, vmin = 0,
        #                             layout=slayout, write_function = self.set_dataset)
        self.frame_index = Settings('frame', dtype = int, initial=55, vmin = 0,
                                    layout=slayout)
        
    
    def create_Buttons(self,blayout):    
        
        self.showWF = Settings('Show WF', dtype=bool, initial=False,
                                     layout=blayout, 
                                     )
        self.showXcorr = Settings('Show Xcorr', dtype=bool, initial=False,
                                     layout=blayout, 
                                     )
        self.showSpectrum = Settings('Show Spectrum', dtype=bool, initial=True,
                                     layout=blayout, 
                                     )
        
        self.keep_calibrating = Settings('Continuos Calibration', dtype=bool, initial=False,
                                     layout=blayout, 
                                     write_function = self.setReconstructor)
        
        buttons_dict = {'Reset': self.reset_processor,
                        'Load calibration': self.loadCalibrationResults,
                        'Calibrate': self.calibration,
                        'Plot calibration phases':self.find_phaseshifts,
                        'SIM reconstruction': self.standard_reconstruction,
                        'Stack SIM reconstruction': self.batch_recontruction,
                        'Stack demodulation': self.stack_demodulation,
                        'Resolution estimation':self.estimate_resolution}

        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            blayout.addWidget(button) 
            
    
    def show_image(self, image_values, basename, name='', scale = 1):
        
        fullname = basename + '_' + name 
        try:
            self.viewer.layers[fullname].data = image_values
        except:
            self.viewer.add_image(image_values, name = fullname, scale = [scale]*image_values.ndim)
        
            
    def select_layer(self, image: Image):
        #selected_layer = list(viewer.layers.selection)[0]
        if hasattr(image,'data'): 
            self.imageRaw_name = image.name
            self.imageRaw = image.data
            
    def select_index(self,val):
        self.frame_index.val = int(viewer.dims.current_step[1])
        self.calculate_WF_image()
        self.calculate_spectrum() # calculates the power spectrum of imageRaw in one of its phasse images
        self.calculate_xcorr() 
        if self.keep_calibrating.val:
            self.calibration()
    
    def get_current_image(self):
        if hasattr(self, 'imageRaw'):    
            return self.imageRaw[:,self.frame_index.val,...]
    
    def start_sim_processor(self):
        self.isCalibrated = False
        self.kx_input = np.zeros((3, 1), dtype=np.single)
        self.ky_input = np.zeros((3, 1), dtype=np.single)
        self.p_input = np.zeros((3, 1), dtype=np.single)
        self.ampl_input = np.zeros((3, 1), dtype=np.single)
        
        if hasattr(self, 'h'):
            self.reset()
            self.start_sim_processor()
        else:
            if self.phases_number.val == 7: 
                self.h = HexSimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor()        
            elif self.phases_number.val == 3:
                self.h = SimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor()           
            else: 
                raise(ValueError("Invalid number of phases"))
            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')
  
    def reset_processor(self,*args):
        self.isCalibrated = False
        self.stop_sim_processor()
        self.start_sim_processor()
        
                
    def calculate_WF_image(self):
        if self.showWF.val:
            img = self.get_current_image()
            imageWF = np.mean(img, axis=0)
            self.show_image(imageWF, 'WF' , self.imageRaw_name)
        
    def calculate_spectrum(self):
        """
        Calculates power spectrum of the image
        """
        if self.showSpectrum.val:
            phase_index = int(viewer.dims.current_step[0])
            img = self.get_current_image()[phase_index,...]
            epsilon = 1e-6
            ps = np.log((np.abs(fftshift(fft2(img))))**2+epsilon)
            self.show_image(ps, 'Spectrum' , self.imageRaw_name)
            
    def calculate_xcorr(self):
        """
        Calculates the crosscorrelation of the low and high pass filtered version of the raw image
        """
        if self.showXcorr.val:
            img = self.get_current_image()
            N = len(img[0, ...])
            _kr, _dk = self.calculate_kr(N)
            M = np.exp(1j * 2 * np.pi / 3) ** ((np.arange(0, 2)[:, np.newaxis]) * np.arange(0, 3))
    
            sum_prepared_comp = np.zeros((2, N, N), dtype=np.complex64)
            
            for k in range(0, 2):
                for l in range(0, 3):
                    sum_prepared_comp[k, ...] = sum_prepared_comp[k, ...] + img[l, ...] * M[k, l]
            
            band0 = sum_prepared_comp[0, ...]
            band1 = sum_prepared_comp[1, ...]
            
            otf_exclude_min_radius = self.h.eta/2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            maskhpf = fftshift(_kr > otf_exclude_min_radius)
            
            band0_common = ifft2(fft2(band0)*maskhpf)
            # band1_common = ifft2(fft2(band1)*maskhpf)
            ix = band0_common * band1
            ixf = np.abs(fftshift(fft2(fftshift(ix))))
            pyc0, pxc0 = self.h._findPeak(ixf )
            #self.add_point( (pyc0, pxc0), 'in calculate carrier' )
            self.show_image(ixf,  'Xcorr' , self.imageRaw_name)
            #self.plot_carrier()
     
    def calculate_kr(self,N):       
        _dx = self.h.pixelsize / self.h.magnification  # Sampling in image plane
        _res = self.h.wavelength / (2 * self.h.NA)
        _oversampling = _res / _dx
        _dk = _oversampling / (N / 2)  # Sampling in frequency plane
        _k = np.arange(-_dk * N / 2, _dk * N / 2, _dk, dtype=np.double)
        #_dx2 = _dx / 2
        _kr = np.sqrt(_k ** 2 + _k[:,np.newaxis] ** 2, dtype=np.single)
        return  _kr, _dk    
      
    def plot_carrier(self): 
        
        if self.isCalibrated:
            kxs = self.h.kx
            kys = self.h.ky
            N = self.h.N
            from collections.abc import Iterable
            if not isinstance(kxs, Iterable):
                kxs = [kxs]
                kys = [kys]
            _kr, _dk = self.calculate_kr(N)
            for idx, (kx,ky) in enumerate(zip(kxs,kys)):
                pxc0 = -kx / _dk + N/2
                pyc0 = -ky / _dk + N/2
                self.add_point( (pyc0, pxc0), str(idx), color = 'red')
    
    def add_point(self, location, name = '', color = 'green'):
        radius = 10
        fullname = f'Carrier_{self.imageRaw_name}_{name}'
        try:
            self.viewer.layers[fullname].data = location
        except:
            self.viewer.add_points(location, size= radius,
                              face_color= [1,1,1,0], name = fullname , 
                              edge_width=0.5, edge_color=color)   
     
    @add_timer
    def stack_demodulation(self): 
        hyperstack = self.imageRaw
        p,z,y,x = hyperstack.shape
        demodulated = np.zeros([z,y,x]).astype('complex64')
        for frame_index in range(z): 
            for p_idx in range(p):
                demodulated[frame_index,:,:] += 2/p * hyperstack[p_idx,frame_index,:,:]*np.exp(1j*2*np.pi*p_idx/p)
        demodulated_abs = np.abs(demodulated).astype('float') 
        self.imageDem = demodulated_abs
        self.show_image(demodulated_abs, 'Demodulated', self.imageRaw_name)
          
    #@add_timer
    def calibration(self):  
        if hasattr(self, 'h') and hasattr(self, 'imageRaw'):
            # self.setReconstructor()
            selected_imRaw = self.get_current_image()
            if self.gpu.val:
                self.h.calibrate_cupy(selected_imRaw, self.find_carrier.val)       
            else:
                self.h.calibrate(selected_imRaw,self.find_carrier.val)          
            self.isCalibrated = True
            self.plot_carrier()
            if self.showXcorr.val:
                self.calculate_xcorr()
            if self.showSpectrum.val:
                self.calculate_spectrum()
             
    @add_timer  
    def standard_reconstruction(self):
        
        current_imageRaw = self.get_current_image()
        if self.isCalibrated:
                
            if self.gpu.val:
                imageSIM = self.h.reconstruct_cupy(current_imageRaw)
    
            elif not self.gpu.val:
                imageSIM = self.h.reconstruct_rfftw(current_imageRaw)
            
            self.imageSIM = imageSIM
            self.show_image(imageSIM,'SIM',self.imageRaw_name, scale = 0.5)
        else:
            warnings.warn('SIM processor not calibrated')
          
    #@add_timer    
    def batch_recontruction(self): # TODO fix this reconstruction with  multiple batches (multiple planes)
        self.setReconstructor()
        if self.isCalibrated:
            # Batch reconstruction
            if self.gpu.val:
                if self.compact.val:
                    imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                elif not self.compact.val:
                    imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)

            elif not self.gpu.val:
                if self.compact.val:
                    imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                elif not self.compact.val:
                    imageSIM = self.h.batchreconstruct(self.imageRaw)
            
        elif not self.isCalibrated:
            nStack = len(self.imageRaw)
            # calibrate & reconstruction
            if self.gpu.val:
                self.h.calibrate_cupy(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
                self.isCalibrated = True
                
                if self.compact.val:
                    imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                elif not self.compact.val:
                    imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
                

            elif not self.gpu.val:
                self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
                self.isCalibrated = True
                
                if self.compact.val:
                    imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                elif not self.compact.val:
                    imageSIM = self.h.batchreconstruct(self.imageRaw)
        self.imageSIM = imageSIM
        self.show_image(imageSIM,'SIM',self.imageRaw_name, scale = 0.5)

        
    def setReconstructor(self,*args):
        
        self.h.usePhases = self.use_phases.val
        self.h.debug = self.debug.val
        self.h.cleanup = self.cleanup.val
        self.h.axial = self.axial.val
        self.h.usemodulation = self.usemodulation.val
        self.h.magnification = self.magnification.val
        self.h.NA = self.NA.val
        self.h.n = self.n.val
        self.h.wavelength = self.wavelength.val
        self.h.pixelsize = self.pixelsize.val
        self.h.alpha = self.alpha.val
        self.h.beta = self.beta.val
        self.h.w = self.w.val
        self.h.eta = self.eta.val
        if not self.find_carrier.val:
            self.h.kx = self.kx_input
            self.h.ky = self.ky_input
        self.calibration()
        

    @add_timer   
    def estimate_resolution(self): #TODO : consider to add QT timers
            pixelsizeWF = self.h.pixelsize / self.h.magnification
            ciWF = ImageDecorr(self.imageWF, square_crop=True,pixel_size=pixelsizeWF)
            optimWF, resWF = ciWF.compute_resolution()
            ciSIM = ImageDecorr(self.imageSIM, square_crop=True,pixel_size=pixelsizeWF/2)
            optimSIM, resSIM = ciSIM.compute_resolution()
            txtDisplay = f"Wide field image resolution:\t {ciWF.resolution:.3f} um \
                  \nSIM image resolution:\t {ciSIM.resolution:.3f} um\n"
            self.show_text(txtDisplay)
        
    def loadCalibrationResults(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory=self.app.settings['save_dir'], filter="Text files (*.txt)")
            file = open(filename,'r')
            loadResults = json.loads(file.read())
            self.kx_input = np.asarray(loadResults["kx"])
            self.ky_input = np.asarray(loadResults["ky"])
            self.show_text("Calibration results are loaded.")
        except:
            self.show_text("Calibration results are not loaded.")
            
    def find_phaseshifts(self):       
        if self.phases_number.val == 7:
            self.find_7phaseshifts()
        if self.phases_number.val == 3:
            self.find_3phaseshifts()
        
    
    def find_7phaseshifts(self):    
        self.phaseshift = np.zeros((4,7))
        self.expected_phase = np.zeros((4,7))
        frame_index = self.frame_index.val
    
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i],self.h.ky[i],self.imageRaw[:,frame_index,:,:])
            self.expected_phase[i,:] = np.arange(7) * 2*(i+1) * np.pi / 7
            self.phaseshift[i,:] = np.unwrap(phase - self.expected_phase[i,:]) + self.expected_phase[i,:] - phase[0]
    
        self.phaseshift[3] = self.phaseshift[2]-self.phaseshift[1]-self.phaseshift[0]
        
        self.phasesPlot.clear()
        for idx in range(len(self.phaseshift)):
            self.plot(self.phaseshift[idx])
            self.plot(self.expected_phase[idx])
            
    def find_3phaseshifts(self):
        frame_index = self.frame_index.val
        phase, _ = self.h.find_phase(self.h.kx,self.h.ky,self.imageRaw[:,frame_index,:,:])
        expected_phase = np.arange(0,2*np.pi ,2*np.pi / 3)
        phaseshift= np.unwrap(phase - expected_phase) - phase[0]
        error = phaseshift-expected_phase
        data_to_plot = np.array([expected_phase, phaseshift, error])
        symbols = ['.','o','|']
        legend = ['expected', 'measured', 'error']
        self.plot(data_to_plot, legend, symbols)
        self.show_text(f"\nExpected phases: {expected_phase}\
                         \nMeasured phases: {phaseshift}\
                         \nError          : {error}\n")
    
    def plot(self, data, legend, symbols):
        import matplotlib.pyplot as plt
        data = np.array(data)
        char_size = 10
        
        plt.rc('font', family='calibri', size=char_size)
        fig = plt.figure(figsize=(4,3), dpi=150)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_title(title, size=char_size)   
        ax.set_xlabel('step', size=char_size)
        ax.set_ylabel('phase', size=char_size)
        
        for cidx,column in enumerate(data):
            marker= symbols[cidx]
            linewidth = 0 if marker == 'o' else 0.8
            ax.plot(column, marker=marker, linewidth =linewidth)    
        ax.xaxis.set_tick_params(labelsize=char_size*0.75)
        ax.yaxis.set_tick_params(labelsize=char_size*0.75)
        ax.legend(legend, loc='best', frameon = False, fontsize=char_size*0.8)
        ax.grid(True, which='major', axis='both', alpha=0.2)
        ph_num = self.phases_number.val
        
        ticks = np.linspace(0, 2*np.pi*(ph_num-1)/ph_num, 2*ph_num-1 )
        
        ax.set_yticks(ticks)
        fig.tight_layout()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    
    def show_text(self, text):
        print(text)    
        
    def showCalibrationTable(self):
        if self.phases_number.val == 3:
            self.show3CalibrationTable()
        elif self.phases_number.val == 7:
            self.show7CalibrationTable()
    
    def show3CalibrationTable(self,tlayout):
        def table_item(element):
            return QTableWidgetItem(str(element).lstrip('[').rstrip(']'))
        
        table = QTableWidget()
        table.setColumnCount(2)
        table.setRowCount(6)
        tlayout.addLayout(table)
        table.setItem(0, 0, table_item('[kx_in]')) 
        table.setItem(0, 1, table_item(self.kx_input[0]))
        
        table.setItem(1, 0, table_item('[ky_in]'))              
        table.setItem(1, 1, table_item(self.ky_input[0]))
        
        table.setItem(2, 0, table_item('[kx]'))             
        table.setItem(2, 1, table_item(self.h.kx))
        
        #
        table.setItem(3, 0, table_item('[ky]'))              
        table.setItem(3, 1, table_item(self.h.ky))
        
        #
        table.setItem(4, 0, table_item('[phase]'))  
        table.setItem(4, 1, table_item(self.h.p))
        
        #
        table.setItem(5, 0, table_item('[amplitude]'))  
        table.setItem(5, 1, table_item(self.h.ampl))
          
    
    def show7CalibrationTable(self, tlayout):
        def table_item(element):
            return QTableWidgetItem(str(element).lstrip('[').rstrip(']'))
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setRowCount(6)
        tlayout.addLayout(table)
        
        table.setItem(0, 0, table_item('[kx_in]'))
        table.setItem(0, 1, table_item(self.kx_input[0])) 
        table.setItem(0, 2, table_item(self.kx_input[1]))
        table.setItem(0, 3, table_item(self.kx_input[2]))
        
        table.setItem(1, 0, table_item('[ky_in]'))             
        table.setItem(1, 1, table_item(self.ky_input[0]))
        table.setItem(1, 2, table_item(self.ky_input[1]))
        table.setItem(1, 3, table_item(self.ky_input[2]))

        table.setItem(2, 0, table_item('[kx]'))             
        table.setItem(2, 1, table_item(self.h.kx[0]))
        table.setItem(2, 2, table_item(self.h.kx[1]))
        table.setItem(2, 3, table_item(self.h.kx[2]))
        #
        table.setItem(3, 0, table_item('[ky]'))              
        table.setItem(3, 1, table_item(self.h.ky[0]))
        table.setItem(3, 2, table_item(self.h.ky[1]))
        table.setItem(3, 3, table_item(self.h.ky[2]))
        #
        table.setItem(4, 0, table_item('[phase]'))  
        table.setItem(4, 1, table_item(self.h.p[0]))
        table.setItem(4, 2, table_item(self.h.p[1]))
        table.setItem(4, 3, table_item(self.h.p[2]))
        #
        table.setItem(5, 0, table_item('[amplitude]'))  
        table.setItem(5, 1, table_item(self.h.ampl[0]))
        table.setItem(5, 2, table_item(self.h.ampl[1]))
        table.setItem(5, 3, table_item(self.h.ampl[2]))      


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

       
if __name__ == '__main__':
    file = 'testSIM.tif'
    viewer = napari.Viewer()
    viewer.open(file)
    widget = HexSimAnalysis(viewer)
    gui = magicgui(widget.select_layer, auto_call=True)
    viewer.window.add_dock_widget(gui,
                                  name = 'Image selection',
                                  add_vertical_stretch = True)
    viewer.window.add_dock_widget(widget,
                                  name = 'HexSim analyzer @Polimi',
                                  add_vertical_stretch = True)
    
    napari.run()      