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

MYPATH ='C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\DatasetTestNapari\\220114_113154_PROCHIP_SIM_ROI.h5'

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
        operations_layout = QVBoxLayout()
        add_section(operations_layout,'Operations')
        layout.addLayout(operations_layout)
        self.create_Operations(operations_layout)
             
        
    def create_Settings(self, slayout): 
        
        self.phases_number = Settings('phases', dtype=int, initial=3, layout=slayout, 
                              write_function = self.reset_processor)
        
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
        self.dz = Settings('dz', dtype=float, initial=0.55, layout=slayout,
                                  spinbox_decimals=2, unit = 'um',
                                  write_function = self.rescaleZ)
        self.alpha = Settings('alpha', dtype=float, initial=0.350,  spinbox_decimals=2, 
                              layout=slayout, write_function = self.setReconstructor)
        self.beta = Settings('beta', dtype=float, initial=0.980, spinbox_step=0.01, 
                             layout=slayout,  spinbox_decimals=3,
                             write_function = self.setReconstructor)
        self.w = Settings('w', dtype=float, initial=2.00, layout=slayout,
                              spinbox_decimals=2,
                              write_function = self.setReconstructor)
        self.eta = Settings('eta', dtype=float, initial=0.34,
                            layout=slayout, spinbox_decimals=3, spinbox_step=0.01,
                            write_function = self.setReconstructor)
        self.use_phases = Settings('use_phases', dtype=bool, initial=True, layout=slayout,                         
                                   write_function = self.setReconstructor)
        self.find_carrier = Settings('Find Carrier', dtype=bool, initial=True,
                                     layout=slayout, 
                                     write_function = self.setReconstructor)
        
        # self.cleanup = Settings('cleanup', dtype=bool, initial=True, layout=slayout, 
        #                   write_function = self.setReconstructor)
        self.usemodulation = Settings('usemodulation', dtype=bool, initial=False,
                                      layout=slayout, write_function = self.setReconstructor)
        self.axial = Settings('axial', dtype=bool, initial=False, layout=slayout, 
                          write_function = self.setReconstructor) 
        self.group = Settings('group', dtype=int, initial=10, vmin=2,
                            layout=slayout,
                            write_function = self.setReconstructor)
        self.frame_index = Settings('frame', dtype = int, initial=0, vmin = 0,
                                    layout=slayout) # remove frame_index


    def create_Operations(self,blayout):    
        
        self.showXcorr = Settings('Show Xcorr', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.setReconstructor
                                     )
        self.showSpectrum = Settings('Show Spectrum', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.setReconstructor
                                     )
        self.showCarrier = Settings('Show Carrier', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.setReconstructor
                                     )
        
        self.keep_calibrating = Settings('Continuos Calibration', dtype=bool, initial=False,
                                     layout=blayout, 
                                     write_function = self.setReconstructor)
        self.keep_reconstructing = Settings('Continuos Reconstruction', dtype=bool, initial=False,
                                     layout=blayout, 
                                     write_function = self.setReconstructor)
        self.batch = Settings('Batch Reconstruction', dtype=bool, initial=False,
                                     layout=blayout, 
                                     write_function = self.setReconstructor)
        self.debug = Settings('debug', dtype=bool, initial=False, layout=blayout,
                          write_function = self.setReconstructor) 
        # self.gpu = Settings('gpu', dtype=bool, initial=False, layout=blayout, 
        #                   write_function = self.setReconstructor) 
        self.compact = Settings('compact', dtype=bool, initial=False, layout=blayout, 
                          write_function = self.setReconstructor) 
        
        buttons_dict = {'Reset': self.reset_processor,
                        'Widefield': self.calculate_WF_image,
                        'Calibrate': self.calibration,
                        'Plot calibration phases':self.find_phaseshifts,
                        'SIM reconstruction': self.standard_reconstruction,
                        'Stack SIM reconstruction': self.stack_reconstruction,
                        'Stack demodulation': self.stack_demodulation,
                        }

        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            blayout.addWidget(button) 
    
    
    def open_h5_dataset(self, path: pathlib.Path = MYPATH,
                        dataset:int = 50 ):
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
        im_layer = self.show_image(stack, fullname=fullname)
        self.rescaleZ()
        self.center_image(im_layer)
        self.viewer.dims.axis_labels = ('phase','z','y','x')
            
    def select_layer(self, image: Image):
        
        if image.data.ndim == 4:
            self.imageRaw_name = image.name
            sp,sz,sy,sx = image.data.shape
            assert sy == sx, 'Non-square images are not supported'
            #self.viewer.dims.current_step = (0,sz//2, sy//2, sx//2)
            self.rescaleZ()
            self.center_image(image)
            if not hasattr(self, 'h'): 
                self.start_sim_processor()
            print(f'Selected image layer: {image.name}')
           
            
    def rescaleZ(self):
        
        self.zscaling = self.dz.val /(self.pixelsize.val/self.magnification.val)
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                if layer.ndim >2:
                    scale = layer.scale 
                    scale[-3] = self.zscaling
                    layer.scale = scale
                    
    def center_image(self, layer):
        s = layer.data.shape
        current_step = [si//2 for si in s]
        self.viewer.dims.current_step = current_step
        
           
    def select_index(self, val = 0):
        try:
            if viewer.dims.ndim >2:
                self.frame_index.val = int(self.viewer.dims.current_step[-3])
            self.check_checkboxes()
            if self.keep_calibrating.val:
                self.calibration()
            if self.keep_reconstructing.val:
                self.standard_reconstruction()
        except Exception as e:
                     print(e)
    
    
    def check_checkboxes(self):
        if self.showXcorr.val:
            self.calculate_xcorr()
        if self.showSpectrum.val:
            self.calculate_spectrum()
        if self.showCarrier.val:
            self.plot_carrier()
            
            
    def show_image(self, image_values, fullname, **kwargs):
        if 'scale' in kwargs.keys():    
            scale = kwargs['scale']
        else:
            scale = [1.]*image_values.ndim
        
        if 'hold' in kwargs.keys() and fullname in self.viewer.layers:
            
            self.viewer.layers[fullname].data = image_values
            self.viewer.layers[fullname].scale = scale
        
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = fullname,
                                            scale = scale,
                                            interpolation = 'bilinear')
            return layer
    
    
    def get_imageRaw(self):
        try:
            return self.viewer.layers[self.imageRaw_name].data
        except:
             raise(KeyError('Please select a valid 4D image (phase,z,y,x)'))
    
    
    def get_current_image(self):

        data = self.get_imageRaw()
        return data[:,self.frame_index.val,...]
    
    
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
                 
        
    def calculate_spectrum(self):
        """
        Calculates power spectrum of the image
        """
        if self.showSpectrum.val:
            phase_index = int(self.viewer.dims.current_step[0])
            img = self.get_current_image()[phase_index,...]
            epsilon = 1e-6
            ps = np.log((np.abs(fftshift(fft2(img))))**2+epsilon)
            imname = 'Spectrum_' + self.imageRaw_name
            self.show_image(ps, imname, hold = True)
            
            
    def calculate_xcorr(self):
        """
        Calculates the crosscorrelation of the low and high pass filtered version of the raw image
        """
        if self.showXcorr.val and self.isCalibrated:
            ixf = self.h.ixf
            pyc0, pxc0 = self.h._findPeak(ixf )
            imname = 'Xcorr_' + self.imageRaw_name
            self.show_image(ixf, imname, hold = True)
       
        # if self.showXcorr.val == True:
        #     img = self.get_current_image()
        #     N = len(img[0, ...])
        #     _kr, _dk = self.calculate_kr(N)
        #     M = np.exp(1j * 2 * np.pi / 3) ** ((np.arange(0, 2)[:, np.newaxis]) * np.arange(0, 3))
    
        #     sum_prepared_comp = np.zeros((2, N, N), dtype=np.complex64)
            
        #     for k in range(0, 2):
        #         for l in range(0, 3):
        #             sum_prepared_comp[k, ...] = sum_prepared_comp[k, ...] + img[l, ...] * M[k, l]
            
        #     band0 = sum_prepared_comp[0, ...]
        #     band1 = sum_prepared_comp[1, ...]
            
        #     otf_exclude_min_radius = self.h.eta/2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        #     maskhpf = fftshift(_kr > otf_exclude_min_radius)
            
        #     band0_common = ifft2(fft2(band0)*maskhpf)
        #     # band1_common = ifft2(fft2(band1)*maskhpf)
        #     ix = band0_common * band1
        #     ixf = np.abs(fftshift(fft2(fftshift(ix))))
        #     pyc0, pxc0 = self.h._findPeak(ixf )
        #     imname = 'Xcorr_' + self.imageRaw_name
        #     self.show_image(ixf, imname, hold = True)
            
     
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
            print('plot carrier executed')
            kxs = self.h.kx
            kys = self.h.ky
            N = self.h.N
            _kr, _dk = self.calculate_kr(N)
            for idx, (kx,ky) in enumerate(zip(kxs,kys)):
                pxc0 = kx[idx] / _dk + N/2
                pyc0 = ky[idx] / _dk + N/2
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
            
     
    def stack_demodulation(self): 
        hyperstack = self.get_imageRaw()
        p,z,y,x = hyperstack.shape
        demodulated = np.zeros([z,y,x]).astype('complex64')
        for frame_index in range(z): 
            for p_idx in range(p):
                demodulated[frame_index,:,:] += 2/p * hyperstack[p_idx,frame_index,:,:]*np.exp(1j*2*np.pi*p_idx/p)
        demodulated_abs = np.abs(demodulated).astype('float') 
        imname = 'Demodulated_' + self.imageRaw_name
        scale = [self.zscaling,1,1]
        self.show_image(demodulated_abs, imname, scale= scale)
        sz,sy,sx = demodulated_abs.shape
        self.viewer.dims.current_step = (sz//2,sy//2,sx//2) #centered in the 3 projections
        print('Stack demodulation completed')
        
    
    def calculate_WF_image(self):
        imageWFdata = np.mean(self.get_imageRaw(), axis=0)
        imname = 'WF_' + self.imageRaw_name
        self.show_image(imageWFdata, imname, scale = [self.zscaling,1,1])
        sz,sy,sx = imageWFdata.shape
        self.viewer.dims.current_step = (sz//2,sy//2,sx//2) #centered in the 3 projections

    
    def calibration(self):  
        if hasattr(self, 'h'):
            data = self.get_imageRaw()
            sp,sz,sy,sx = data.shape
            idx = self.frame_index.val
            delta = self.group.val // 2
            remainer = self.group.val % 2
            zmin = max(idx-delta,0)
            zmax = min(idx+delta+remainer,sz)
            new_delta = zmax-zmin
            data = data[:,zmin:zmax,:,:]

            selected_imRaw = np.swapaxes(data, 0, 1).reshape((sp * new_delta, sy, sx))
            
            self.h.calibrate(selected_imRaw,self.find_carrier.val)          
            self.isCalibrated = True
            #self.check_checkboxes()
            if self.find_carrier.val: # store the value found   
                self.kx_input = self.h.kx  
                self.ky_input = self.h.ky
                self.p_input = self.h.p
                self.ampl_input = self.h.ampl      
             
    
    def standard_reconstruction(self):  
        current_imageRaw = self.get_current_image()
        if self.isCalibrated:
            print('recon executed')   
            
            imageSIM = self.h.reconstruct_rfftw(current_imageRaw)
            
            imname = 'SIM_' + self.imageRaw_name
            self.show_image(imageSIM, fullname=imname, scale=[0.5,0.5], hold =True)
                
        else:
            raise(Warning('SIM processor not calibrated'))  
            
    def stack_reconstruction(self):
        
        def update_sim_image(stack):
            imname = 'SIMstack_' + self.imageRaw_name
            scale = [self.zscaling,0.5,0.5]
            self.show_image(stack, fullname=imname, scale=scale)
                
            sz,sy,sx = stack.shape
            self.viewer.dims.current_step = (sz//2,sy//2,sx//2) #centered in the 3 projections
            print('Stack reconstruction completed')
        
        @thread_worker(connect={'returned': update_sim_image})
        def _stack_reconstruction():
            import warnings
            warnings.filterwarnings('ignore')
            hyperstack = self.get_imageRaw()
            sp,sz,sy,sx = hyperstack.shape
            stackSIM = np.zeros([sz,2*sy,2*sx], dtype=np.single)
            for zidx in range(sz):
                phases_stack = hyperstack[:,zidx,:,:]
                if self.keep_calibrating.val:
                    delta = self.group.val // 2
                    remainer = self.group.val % 2
                    zmin = max(zidx-delta,0)
                    zmax = min(zidx+delta+remainer,sz)
                    new_delta = zmax-zmin
                    data = hyperstack[:,zmin:zmax,:,:]
                    selected_imRaw = np.swapaxes(data, 0, 1).reshape((sp * new_delta, sy, sx))
                    self.h.calibrate(selected_imRaw,self.find_carrier.val)   
 
                stackSIM[zidx,:,:] = self.h.reconstruct_rfftw(phases_stack)
            return stackSIM
        
        @thread_worker(connect={'returned': update_sim_image})
        def _batch_reconstruction():
            import warnings
            warnings.filterwarnings('ignore')
            hyperstack = self.get_imageRaw()
            sp,sz,sy,sx = hyperstack.shape
            hyperstack = np.swapaxes(hyperstack, 0, 1).reshape((sp * sz, sy, sx))
            if self.compact.val:
                stackSIM = self.h.batchreconstructcompact(hyperstack)
            elif not self.compact.val:
                stackSIM = self.h.batchreconstruct(hyperstack)
            return stackSIM
        
        # main function exetuted here
        if not self.isCalibrated:
            raise(Warning('SIM processor not calibrated'))  
            return
        else:
            if self.batch.val:
                _batch_reconstruction()
            else: 
                _stack_reconstruction()
                
        
    def estimate_resolution(self, image:Image):
        from image_decorr import ImageDecorr
        @thread_worker
        def _estimate_resolution():
            import warnings
            warnings.filterwarnings('ignore')
            pixelsize = self.h.pixelsize / self.h.magnification
            dims = image.data.ndim
            if dims == 2:
                im = image.data
            elif dims ==3:
                im = image.data[self.frame_index.val, :,:]
            else:
                raise(TypeError(f'Resolution estimation not supported for {dims} dimensional data'))
            scalex = image.scale[-1]
            ci = ImageDecorr(im, square_crop=True, pixel_size=pixelsize*scalex)
            optim, res = ci.compute_resolution()
            txtDisplay = f"Image resolution: {ci.resolution:.3f} um"
            print(txtDisplay)
        worker = _estimate_resolution()
        worker.start()
        
          
    def find_phaseshifts(self):
        if self.isCalibrated:
            if self.phases_number.val == 7:
                self.find_7phaseshifts()
            if self.phases_number.val == 3:
                self.find_3phaseshifts()
            self.showCalibrationTable() 
        else:
            raise(Warning('SIM processor not calibrated, unable to show phases'))
            
        
    def find_7phaseshifts(self):    
        phaseshift = np.zeros((4,7))
        expected_phase = np.zeros((4,7))
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i],self.h.ky[i],self.get_current_image())
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
        phase, _ = self.h.find_phase(self.h.kx,self.h.ky,self.get_current_image())
        expected_phase = np.arange(0,2*np.pi ,2*np.pi / 3)
        #phaseshift= np.unwrap(phase - expected_phase) - phase[0]
        phaseshift = np.unwrap(phase) - phase[0]
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
    
    def make_layers_visible(self, *layers_list):
        
        for layer in self.viewer.layers:
            if layer in layers_list:
                layer.visible = True
            else:
                layer.visible = False    
        
    def register_stack(self,image:Image, mode='Euclidean'):
    
        def add_image(data):
            self.viewer.add_image(data, 
                                scale = image.scale,
                                interpolation = 'bilinear',
                                name = f'registered_{image.name}')
            print('Registration completed')
        
        @thread_worker(connect={'returned': add_image})
        def _register_stack():
            import warnings
            warnings.filterwarnings('ignore')
            frame_idx = self.frame_index.val
            stack = image.data
            registered = stack_registration(stack, z_idx=frame_idx, c_idx=0, method = 'cv2', mode=mode)
            return registered
            
        _register_stack() 
    
    def segment(self,
                image_layer:Image,
                find_thresold:bool = True,
                thresold:int = 0,
                median_filter_size:int = 2,
                erosion_size:int = 4,
                remove_smaller_than:int = 6,
                fill_holes:bool = True,
                smooth_size:int = 3,
                show_labels:bool = True                
                ):
        
        from scipy import ndimage as ndi
        from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_local, threshold_mean
        from skimage.segmentation import clear_border
        from skimage.measure import label
        from skimage.morphology import erosion, dilation, closing, opening, cube, ball, remove_small_objects, remove_small_holes
   
        def add_labels(data):
            label_name = f'segmentation_{image_layer.name}'
            
            if  label_name in self.viewer.layers:
                self.viewer.layers[label_name].data = data 
            else:  
                self.viewer.add_labels(data, 
                                scale = image_layer.scale,
                                name = label_name)
            
            self.make_layers_visible(self.viewer.layers[label_name], image_layer)
            self.viewer.layers.selection = [image_layer]
            #self.viewer.scale_bar.visible = True
            print('Segmentation completed')
        
        @thread_worker(connect={'returned': add_labels})
        def _segment():
            import warnings
            warnings.filterwarnings('ignore')
            
            if image_layer.ndim == 4:
                
                #current_step = self.viewer.dims.current_step
                data= np.array(image_layer.data)[0,...]
            
            elif image_layer.ndim ==3:
            # #     #current_step = self.viewer.dims.current_step
                data= np.array(image_layer.data)
            else:
                 raise(ValueError('image layer dimensionality not supported'))
            
            if find_thresold:
               thres = threshold_otsu(data)
               
               segment_widget.thresold.value = thres   
            else:
                thres = thresold
            # print(f'{thres=}')
            bw = data > thres
            #bw = clear_border(bw)
            
            if erosion_size > 0:
                structure = ball(erosion_size)
                ssz,ssy,ssx = structure.shape
                im_scale = image_layer.scale
                structure = np.resize(structure, [max(int(im_scale[2]/im_scale[1]*ssz),1), ssy, ssx])
                bw = erosion(bw, structure)
            
            if remove_smaller_than > 0:
                bw = remove_small_objects(bw, min_size=remove_smaller_than**3)
            
            if fill_holes:
                bw = ndi.binary_fill_holes(bw)
                
            if smooth_size>0:
                sphere= ball(smooth_size)
                bw = dilation(bw, sphere)
                bw = erosion(bw, sphere)
            
            if show_labels:
                labels_stack = label(bw)
            else:
                labels_stack = bw
            
            return (labels_stack)    
    
        _segment()

       
if __name__ == '__main__':
    file = 'test.tif'
    viewer = napari.Viewer()
    #viewer.open(file)
    widget = HexSimAnalysis(viewer)
    mode={"choices": ['Translation','Affine','Euclidean','Homography']}
    registration = magicgui(widget.register_stack, call_button='Register stack', mode=mode)
    selection = magicgui(widget.select_layer, auto_call=True )# call_button='Select image layer')
    h5_opener = magicgui(widget.open_h5_dataset, call_button='Open h5 dataset')
    resolution = magicgui(widget.estimate_resolution, call_button='Estimate resolution')
    # segment_widget = magicgui(widget.segment, call_button='Run segmentation')
    
    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(selection,
                                  name = 'Image layer selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(widget,
                                  name = 'HexSim analyzer @Polimi',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(registration,
                                  name = 'Stack registration',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(resolution,
                                  name = 'Resolution estimation',
                                  add_vertical_stretch = True)
    # segment_widget.thresold.max = 2**16
    # viewer.window.add_dock_widget(segment_widget,
    #                               name = '3D segmentation',
    #                               add_vertical_stretch = True)
    
    napari.run()      