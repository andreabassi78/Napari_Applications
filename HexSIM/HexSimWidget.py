# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:34:41 2022

@author: Andrea Bassi @Polimi
"""
import napari

from qtpy.QtWidgets import QTableWidget, QSplitter, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QComboBox, QWidget, QFrame, QLabel, QFormLayout, QVBoxLayout, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox
from skimage.measure import regionprops
from napari.layers import Image, Points, Labels, Shapes
import numpy as np
from numpy.fft import ifft2, fft2, fftshift
from qtpy.QtWidgets import QFileDialog, QTableWidgetItem   

from hexSimProcessor import HexSimProcessor
from simProcessor import SimProcessor 
from image_decorr import ImageDecorr
from magicgui import magicgui
from widget_settings import Settings, add_timer
import warnings
import pathlib

class HexSimAnalysis(QWidget):
    
    name = 'HexSIM_Analysis'

    def __init__(self, viewer:napari.Viewer,
                 ):
        self.viewer = viewer
        super().__init__()
        
        self.setup_ui() # run setup_ui before instanciating the Settings
        self.start_sim_processor()
        self.viewer.dims.events.current_step.connect(self.select_index)
        
    def setup_ui(self):     
        
        def add_section(_layout,_title):
            from qtpy.QtCore import Qt
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
        self.gpu = Settings('gpu', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.compact = Settings('compact', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
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
        self.dz = Settings('dz', dtype=float, initial=0.25, layout=slayout,
                                  spinbox_decimals=2, unit = 'um',
                                  write_function = self.rescale)
        self.alpha = Settings('alpha', dtype=float, initial=0.350,  spinbox_decimals=2, 
                              layout=slayout, write_function = self.setReconstructor)
        self.beta = Settings('beta', dtype=float, initial=0.980, spinbox_step=0.01, 
                             layout=slayout,  spinbox_decimals=3,
                             write_function = self.setReconstructor)
        self.w = Settings('w', dtype=float, initial=2.00, layout=slayout,
                              spinbox_decimals=2,
                              write_function = self.setReconstructor)
        self.eta = Settings('eta', dtype=float, initial=0.4,
                            layout=slayout, spinbox_decimals=3, spinbox_step=0.05,
                            write_function = self.setReconstructor)
        self.use_phases = Settings('use_phases', dtype=bool, initial=True, layout=slayout,                         
                                   write_function = self.setReconstructor)
        self.find_carrier = Settings('Find Carrier', dtype=bool, initial=True,
                                     layout=slayout, 
                                     write_function = self.setReconstructor)
        self.cleanup = Settings('cleanup', dtype=bool, initial=True, layout=slayout, 
                          write_function = self.setReconstructor)
        self.usemodulation = Settings('usemodulation', dtype=bool, initial=False,
                                      layout=slayout, write_function = self.setReconstructor)
        self.axial = Settings('axial', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.frame_index = Settings('frame', dtype = int, initial=0, vmin = 0,
                                    layout=slayout)


    def create_Buttons(self,blayout):    
        
        self.showXcorr = Settings('Show Xcorr', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.setReconstructor
                                     )
        self.showSpectrum = Settings('Show Spectrum', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.setReconstructor
                                     )
        self.showCarrier = Settings('Show Carrier', dtype=bool, initial=True,
                                     layout=blayout,
                                     write_function = self.setReconstructor
                                     )
        
        self.keep_calibrating = Settings('Continuos Calibration', dtype=bool, initial=False,
                                     layout=blayout, 
                                     write_function = self.setReconstructor)
        self.keep_reconstructing = Settings('Continuos Reconstruction', dtype=bool, initial=False,
                                     layout=blayout, 
                                     write_function = self.setReconstructor)
        buttons_dict = {'Reset': self.reset_processor,
                        'Widefield': self.calculate_WF_image,
                        'Calibrate': self.calibration,
                        'Plot calibration phases':self.find_phaseshifts,
                        'SIM reconstruction': self.standard_reconstruction,
                        'Stack SIM reconstruction': self.stack_reconstruction,
                        'Stack demodulation': self.stack_demodulation,
                        #'Rescale': self.rescale,
                        'Resolution estimation':self.estimate_resolution}

        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            blayout.addWidget(button) 
            
    
    def open_h5_dataset(self, path: pathlib.Path = "test.h5",
                        dataset:int = 0 )->Image:
                
        # open file
        from get_h5_data import get_h5_dataset, get_multiple_h5_datasets, get_h5_attr, get_datasets_index_by_name, get_group_name
        import os
        directory, filename = os.path.split(path)
        t_idx = f'/t{dataset:04d}/'
        index_list, names = get_datasets_index_by_name(path, t_idx)
        stack,found = get_multiple_h5_datasets(path, index_list)
        sp,sz,sy,sx = stack.shape
        if sp != 3 and sp != 7:  
            print(f'\nUnable to open dataset {dataset}.\n')
            raise(ValueError)
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
        
        return Image(stack, name = f'dts{dataset}_{filename}') 
      
    
    
    def show_image(self, image_values, fullname, **kwargs):
        try:
            self.viewer.layers[fullname].data = image_values
        except:
            if 'scale' in kwargs.keys():    
                scale = kwargs['scale']
            else:
                scale = [1.]*image_values.ndim
            self.viewer.add_image(image_values, name = fullname, scale = scale)
            
                 
    def select_layer(self, image: Image):
        if hasattr(image,'data'): 
            self.imageRaw_name = image.name
            self.imageRaw = image.data
            sp,sz,sy,sx = image.data.shape
            assert sy == sx, 'Non-square image shape is not supported'
            self.viewer.dims.current_step = (0,sz//2, sy//2, sx//2)
            self.rescale()
            if not hasattr(self, 'h'): 
                self.start_sim_processor()
                
            
    def rescale(self):
        self.zscaling = self.dz.val /(self.pixelsize.val/self.magnification.val)
        self.viewer.layers[self.imageRaw_name].scale = [1,self.zscaling,1,1]
            
    def select_index(self, val = 0):
        self.frame_index.val = int(self.viewer.dims.current_step[1])
        self.check_checkboxes()
        if self.keep_calibrating.val:
            self.calibration()
        if self.keep_reconstructing.val:
            self.standard_reconstruction()
    
    def check_checkboxes(self):
        if self.showXcorr.val:
            self.calculate_xcorr()
        if self.showSpectrum.val:
            self.calculate_spectrum()
        if self.showCarrier.val:
            self.plot_carrier()

    
    def get_current_image(self):
        if hasattr(self, 'imageRaw'):    
            return self.imageRaw[:,self.frame_index.val,...]
    
    def start_sim_processor(self):     
        self.isCalibrated = False
        
        if hasattr(self, 'h'):
            self.stop_sim_processor()
            self.start_sim_processor()
        else:
            if self.phases_number.val == 7: 
                self.h = HexSimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor() 
                self.kx_input = np.zeros((3, 1), dtype=np.single)
                self.ky_input = np.zeros((3, 1), dtype=np.single)
                self.p_input = np.zeros((3, 1), dtype=np.single)
                self.ampl_input = np.zeros((3, 1), dtype=np.single)
            elif self.phases_number.val == 3:
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
        self.select_index()
        if not self.find_carrier.val:
            self.h.kx = self.kx_input
            self.h.ky = self.ky_input
        if self.keep_calibrating.val:
            self.calibration()
        if self.keep_reconstructing.val:
            self.standard_reconstruction()
                 
    def calculate_WF_image(self):
        if hasattr(self, 'imageRaw'):
            imageWF = np.mean(self.imageRaw, axis=0)
            imname = 'WF_' + self.imageRaw_name
            self.show_image(imageWF, imname, scale = [self.zscaling,1,1])
            self.imageWF = imageWF
        
    def calculate_spectrum(self):
        """
        Calculates power spectrum of the image
        """
        if hasattr(self, 'imageRaw') and self.showSpectrum.val == True:
            phase_index = int(self.viewer.dims.current_step[0])
            img = self.get_current_image()[phase_index,...]
            epsilon = 1e-6
            ps = np.log((np.abs(fftshift(fft2(img))))**2+epsilon)
            imname = 'Spectrum_' + self.imageRaw_name
            self.show_image(ps, imname)
            
    def calculate_xcorr(self):
        """
        Calculates the crosscorrelation of the low and high pass filtered version of the raw image
        """
        if hasattr(self, 'imageRaw') and self.showXcorr.val == True:
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
            imname = 'Xcorr_' + self.imageRaw_name
            self.show_image(ixf, imname)
            
     
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
        
        if self.showCarrier.val and self.isCalibrated:
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
        imname = 'Demodulated_' + self.imageRaw_name
        scale = [self.zscaling,1,1]
        self.show_image(demodulated_abs, imname, scale = scale)
          
        
    def calibration(self):  
        if hasattr(self, 'h') and hasattr(self, 'imageRaw'):
            
            selected_imRaw = self.get_current_image()
            # fidx = self.frame_index.val
            # selected_imRaw = np.mean(self.imageRaw[:,fidx:fidx+10,:,:], axis = 1)
            
            if self.gpu.val:
                self.h.calibrate_cupy(selected_imRaw, self.find_carrier.val)       
            else:
                self.h.calibrate(selected_imRaw,self.find_carrier.val)          
            self.isCalibrated = True
            self.check_checkboxes()
            if self.find_carrier.val: # store the value found   
                self.kx_input = self.h.kx  
                self.ky_input = self.h.ky
                self.p_input = self.h.p
                self.ampl_input = self.h.ampl
             
    
    def standard_reconstruction(self):  
        current_imageRaw = self.get_current_image()
        if self.isCalibrated:
                
            if self.gpu.val:
                imageSIM = self.h.reconstruct_cupy(current_imageRaw)
    
            elif not self.gpu.val:
                imageSIM = self.h.reconstruct_rfftw(current_imageRaw)
            
            self.imageSIM = imageSIM
            imname = 'SIM_' + self.imageRaw_name
            self.show_image(imageSIM, imname, scale = [0.5,0.5])
        else:
            warnings.warn('SIM processor not calibrated')
              
            
    def stack_reconstruction(self):
        from napari.qt.threading import thread_worker
        
        def update_sim_image(stack):
            imname = 'SIMstack_' + self.imageRaw_name
            self.show_image(stack, imname, scale = [self.zscaling,0.5,0.5])
            self.imageSIM = stack
            sz,sy,sx = stack.shape
            self.viewer.dims.current_step = (sz//2,sy//2,sx//2) #centered in the 3 projections
        
        @thread_worker(connect={'returned': update_sim_image})
        def _stack_reconstruction():
            import warnings
            warnings.filterwarnings('ignore')
            hyperstack = self.imageRaw
            sp,sz,sy,sx = hyperstack.shape
            stackSIM = np.zeros([sz,2*sy,2*sx], dtype=np.single)
            for zidx in range(sz):
                phases_stack = hyperstack[:,zidx,:,:]
                if self.keep_calibrating.val:
                    self.h.calibrate(phases_stack, self.find_carrier.val)   
                stackSIM[zidx,:,:] = self.h.reconstruct_rfftw(phases_stack)
            return stackSIM
        
        if not self.isCalibrated:
            warnings.warn('SIM processor not calibrated')
            return
        else:
            _stack_reconstruction()
        
        
    @add_timer    
    def batch_recontruction(self): # TODO fix this reconstruction with  multiple batches (multiple planes)

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
                    
        self.imageSIM = imageSIM
        imname = 'SIM_' + self.imageRaw_name
        scale = [self.zscaling, 0.5, 0.5]
        simlayer =self.show_image(imageSIM, imname, scale = scale)
        
    def estimate_resolution(self): #TODO : consider to add QT timers
        from napari.qt.threading import thread_worker
        
        @thread_worker
        def _estimate_resolution():
            import warnings
            warnings.filterwarnings('ignore')
            pixelsizeWF = self.h.pixelsize / self.h.magnification
            imWF = self.imageWF[self.frame_index.val, ...]
            ciWF = ImageDecorr(imWF, square_crop=True, pixel_size=pixelsizeWF)
            optimWF, resWF = ciWF.compute_resolution()
            if self.imageSIM.ndim >2:
                imsim = self.imageSIM[self.frame_index.val, ...]
            else:
                imsim = self.imageSIM
            ciSIM = ImageDecorr(imsim, square_crop=True,pixel_size=pixelsizeWF/2)
            optimSIM, resSIM = ciSIM.compute_resolution()
            txtDisplay = f"Wide field image resolution:\t {ciWF.resolution:.3f} um \
                        \nSIM image resolution:\t {ciSIM.resolution:.3f} um\n"
            print(txtDisplay)
        worker = _estimate_resolution()  # create "worker" object
        #worker.returned.connect(update_image)  # connect callback functions
        worker.start()
          
    def find_phaseshifts(self):
        if self.isCalibrated:
            if self.phases_number.val == 7:
                self.find_7phaseshifts()
            if self.phases_number.val == 3:
                self.find_3phaseshifts()
            self.showCalibrationTable() 
        else:
            warnings.warn(' Processor not calibrated, unable to show phases')
        
    
    def find_7phaseshifts(self):    
        phaseshift = np.zeros((4,7))
        expected_phase = np.zeros((4,7))
        frame_index = self.frame_index.val
    
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i],self.h.ky[i],self.imageRaw[:,frame_index,:,:])
            expected_phase[i,:] = np.arange(7) * 2*(i+1) * np.pi / 7
            phaseshift[i,:] = np.unwrap(phase - expected_phase[i,:]) + expected_phase[i,:] - phase[0]
    
        phaseshift[3] = phaseshift[2]-phaseshift[1]-phaseshift[0]
        data_list = []
        for idx in range(len(phaseshift)):
            data_list.append([expected_phase[idx],phaseshift[idx]])
        data_to_plot = np.array(data_list)    
        symbols = ['.','o']
        legend = ['expected', 'measured']
        self.plot_with_plt(data_to_plot, legend, symbols)
            
    def find_3phaseshifts(self):
        frame_index = self.frame_index.val
        phase, _ = self.h.find_phase(self.h.kx,self.h.ky,self.imageRaw[:,frame_index,:,:])
        expected_phase = np.arange(0,2*np.pi ,2*np.pi / 3)
        phaseshift= np.unwrap(phase - expected_phase) - phase[0]
        error = phaseshift-expected_phase
        data_to_plot = np.array([expected_phase, phaseshift, error])
        symbols = ['.','o','|']
        legend = ['expected', 'measured', 'error']
        self.plot_with_plt(data_to_plot, legend, symbols)
        print(f"\nExpected phases: {expected_phase}\
                         \nMeasured phases: {phaseshift}\
                         \nError          : {error}\n")
    
    def plot_with_plt(self, data, legend, symbols):
        import matplotlib.pyplot as plt
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
        vales_num = data.shape[1]
        ticks = np.linspace(0, 2*np.pi*(vales_num-1)/vales_num, 2*vales_num-1 )
        ax.set_yticks(ticks)
        fig.tight_layout()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)
    
    def showCalibrationTable(self):
        import pandas as pd
        headers= ['kx_in','ky_in','kx','ky','phase','amplitude']
        vals = [self.kx_input, self.ky_input,
                self.h.kx, self.h.ky,
                self.h.p,self.h.ampl]

        table = pd.DataFrame([vals] , columns = headers )
        print(table)

       
if __name__ == '__main__':
    file = 'test.tif'
    viewer = napari.Viewer()
    viewer.open(file)
    widget = HexSimAnalysis(viewer)
    gui = magicgui(widget.select_layer, auto_call=True)
    h5_opener = magicgui(widget.open_h5_dataset, call_button='Open h5 dataset')
    viewer.window.add_dock_widget((h5_opener,gui),
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    viewer.window.add_dock_widget(gui,
                                  name = 'Image layer selection',
                                  add_vertical_stretch = True)
    viewer.window.add_dock_widget(widget,
                                  name = 'HexSim analyzer @Polimi',
                                  add_vertical_stretch = True)
    
    napari.run()      