# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:34:41 2022

@author: Andrea Bassi @Polimi
"""
import napari
from qtpy.QtWidgets import QLabel, QVBoxLayout,QSplitter, QHBoxLayout, QWidget, QPushButton
from napari.layers import Image
import numpy as np
from numpy.fft import fft2, fftshift 
from napari.qt.threading import thread_worker
from hexSimProcessor import HexSimProcessor
from convSimProcessor import ConvSimProcessor
from simProcessor import SimProcessor 
from magicgui import magicgui
from widget_settings import Settings, add_timer, add_timer_to_function
import warnings
import pathlib
from  registration_tools import stack_registration

#MYPATH ='C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\DatasetTestNapari\\220114_113154_PROCHIP_SIM_ROI.h5'
#MYPATH = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\PROCHIP\\Spyder\\220228\\220228_192448_PROCHIP_HexSIM_ROI.h5'
MYPATH ='C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\DatasetTestNapari\\220218_145747_PROCHIP_SIM_ROI_dts_00_05_30_33.h5'
class HexSimAnalysis(QWidget):
    
    name = 'HexSIM_Analysis'

    def __init__(self, viewer:napari.Viewer,
                 ):
        self.viewer = viewer
        super().__init__()
        
        self.setup_ui() # run setup_ui before instanciating the Settings
        self.start_sim_processor()
        self.viewer.dims.events.current_step.connect(self.on_step_change)
        
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
        self.sim_method = Settings('SIM acquisition mode', dtype=int, initial=2, layout=slayout, 
                              write_function = self.reset_processor)
        self.phases_number = Settings('phases', dtype=int, initial=7, layout=slayout, 
                              write_function = self.reset_processor)
        self.angles_number = Settings('angles', dtype=int, initial=1, layout=slayout, 
                              write_function = self.reset_processor)
        
        self.magnification = Settings('M', dtype=float, initial=100, unit = 'X',  
                                      layout=slayout, write_function = self.setReconstructor)
        self.NA = Settings('NA', dtype=float, initial=1.05, layout=slayout, 
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
        self.alpha = Settings('alpha', dtype=float, initial=0.5,  spinbox_decimals=2, 
                              layout=slayout, write_function = self.setReconstructor)
        self.beta = Settings('beta', dtype=float, initial=0.980, spinbox_step=0.01, 
                             layout=slayout,  spinbox_decimals=3,
                             write_function = self.setReconstructor)
        self.w = Settings('w', dtype=float, initial=0.5, layout=slayout,
                              spinbox_decimals=2,
                              write_function = self.setReconstructor)
        self.eta = Settings('eta', dtype=float, initial=0.75,
                            layout=slayout, spinbox_decimals=3, spinbox_step=0.01,
                            write_function = self.setReconstructor)
        self.use_phases = Settings('use_phases', dtype=bool, initial=True, layout=slayout,                         
                                   write_function = self.setReconstructor)
        self.find_carrier = Settings('Find Carrier', dtype=bool, initial=True,
                                     layout=slayout, 
                                     write_function = self.setReconstructor)
        
        self.group = Settings('group', dtype=int, initial=30, vmin=2,
                            layout=slayout,
                            write_function = self.setReconstructor)


    def create_Operations(self,blayout):    
        
        self.showXcorr = Settings('Show Xcorr', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.show_xcorr
                                     )
        
        self.showSpectrum = Settings('Show Spectrum', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.show_spectrum
                                     )
        self.showCarrier = Settings('Show Carrier', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.show_carrier
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
        self.gpu = Settings('gpu', dtype=bool, initial=False, layout=blayout, 
                           write_function = self.setReconstructor) 
        
        buttons_dict = {'Reset': self.reset_processor,
                        'Widefield': self.calculate_WF_image,
                        'Calibrate': self.calibration,
                        'Plot calibration phases':self.find_phaseshifts,
                        'SIM reconstruction': self.single_plane_reconstruction,
                        'Stack SIM reconstruction': self.stack_reconstruction,
                        'Show Wiener filter':self.show_wiener,
                        'Stack demodulation': self.stack_demodulation,
                        }

        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            blayout.addWidget(button) 
    
    
    def open_h5_dataset(self, path: pathlib.Path = MYPATH,
                        dataset:int = 33 ):
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
        sp,sz,sy,sx = stack.shape
        assert sy == sx, 'Non-square images are not supported'
        fullname = f'dts{dataset}_{filename}'
        im_layer = self.show_image(stack, fullname=fullname)
        self.rescaleZ()
        self.center_image(im_layer)
        self.viewer.dims.axis_labels = ('phase','z','y','x')
            
        
    def select_layer(self, image: Image):
        
        if image.data.ndim == 5:
            self.imageRaw_name = image.name
            sa,sp,sz,sy,sx = image.data.shape
            self.angles_number.val = sa
            self.phases_number.val = sp
            self.viewer.dims.axis_labels = ["angle", "phase", "z", "y","x"]
        elif image.data.ndim == 4:
            self.imageRaw_name = image.name
            sp,sz,sy,sx = image.data.shape
            self.angles_number.val = 1
            self.phases_number.val = sp
            #self.viewer.dims.axis_labels = ["phase", "z", "y","x"]
        else:
            return
            
        assert sy == sx, 'Non-square images are not supported'
        self.rescaleZ()
        self.center_image(image)
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
        
           
    def on_step_change(self, val = 0):   
        if self.viewer.dims.ndim >3:
            self.setReconstructor()
            if self.showSpectrum.val:
                 self.show_spectrum()
            
            
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
        z_index = int(self.viewer.dims.current_step[-3])
        data = self.get_imageRaw()
        return data[...,z_index,:,:]
    
    
    def start_sim_processor(self):     
        self.isCalibrated = False
        
        if hasattr(self, 'h'):
            self.stop_sim_processor()
            self.start_sim_processor()
        else:
            if self.sim_method.val == 0: # insert combo box
                self.h = SimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.h.debug = False
                self.setReconstructor() 
                self.kx_input = np.zeros((1, 1), dtype=np.single)
                self.ky_input = np.zeros((1, 1), dtype=np.single)
                self.p_input = np.zeros((1, 1), dtype=np.single)
                self.ampl_input = np.zeros((1, 1), dtype=np.single)
            
            elif self.sim_method.val == 1: 
                self.h = HexSimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.h.debug = False
                self.setReconstructor() 
                self.kx_input = np.zeros((3, 1), dtype=np.single)
                self.ky_input = np.zeros((3, 1), dtype=np.single)
                self.p_input = np.zeros((3, 1), dtype=np.single)
                self.ampl_input = np.zeros((3, 1), dtype=np.single)
            
            elif self.sim_method.val == 2:
                self.h = ConvSimProcessor()
                self.h.opencv = False
                self.h.debug = False
                self.setReconstructor() 
                self.kx_input = np.zeros((3, 1), dtype=np.single)
                self.ky_input = np.zeros((3, 1), dtype=np.single)
                self.p_input = np.zeros((3, 1), dtype=np.single)
                self.ampl_input = np.zeros((3, 1), dtype=np.single)
                
            else: 
                raise(ValueError("Invalid SIM method"))

            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')
  
    
    def reset_processor(self,*args):
        self.isCalibrated = False
        self.stop_sim_processor()
        self.start_sim_processor()
           
    @add_timer    
    def setReconstructor(self,*args):
        if hasattr(self, 'h'):   
            self.h.usePhases = self.use_phases.val
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
            if self.keep_calibrating.val:
                self.calibration()
            if self.keep_reconstructing.val:
                self.single_plane_reconstruction()
          
            
    def show_wiener(self):
        """
        Shows the wiener filter
        """
        
        if self.isCalibrated:
            imname = 'Wiener_' + self.imageRaw_name
            img = self.h.wienerfilter
            swy,swx = img.shape
            self.show_image(img[swy//2-swy//4:swy//2+swy//4,swx//2-swx//4:swx//2+swx//4],
                            imname, hold = True, scale=[1,1])
        
        
    def show_spectrum(self):
        """
        Calculates power spectrum of the image
        """
        if self.showSpectrum.val and hasattr(self, 'imageRaw_name') and self.viewer.dims.ndim >3:
            imname = 'Spectrum_' + self.imageRaw_name
            phase_index = int(self.viewer.dims.current_step[-4])
            img = self.get_current_image()
            ndims = img.ndim
            if ndims == 4:
                 angle_index = int(self.viewer.dims.current_step[-5])   
                 img0 = self.get_current_image()[angle_index,phase_index,:,:]
            elif ndims == 3:
                img0 = self.get_current_image()[phase_index,:,:]
                
            else:
                return                    
            epsilon = 1e-10
            ps = np.log((np.abs(fftshift(fft2(img0))))**2+epsilon)
            
            self.show_image(ps, imname, hold = True)
            
            
    def show_xcorr(self):
        """
        Show the crosscorrelation of the low and high pass filtered version of the raw images,
        used forfinding the carrier
        """
        if self.showXcorr.val and self.isCalibrated:
            ixf = self.h.ixf
            #ixf_refined = self.h.ixf_refined
            imname = 'Xcorr_' + self.imageRaw_name
            #refined_imname = 'Xcorr_refined_' + self.imageRaw_name
            self.show_image(ixf, imname, hold = True)
            #zoom = self.h.zoom 
            #self.show_image(ixf_refined, refined_imname, hold = True, scale = (zoom,zoom))
            
     
    def calculate_kr(self,N):       
        dx = self.h.pixelsize / self.h.magnification  # Sampling in image plane
        res = self.h.wavelength / (2 * self.h.NA)
        oversampling = res / dx
        dk = oversampling / (N / 2)  # Sampling in frequency plane
        k = np.arange(-dk * N / 2, dk * N / 2, dk, dtype=np.double)
        kr = np.sqrt(k ** 2 + k[:,np.newaxis] ** 2, dtype=np.single)
        return  kr, dk    
      
        
    def show_carrier(self): 
        if self.showCarrier.val and self.isCalibrated:
                kxs = self.h.kx
                kys = self.h.ky
                N = self.h.N
                _kr, dk = self.calculate_kr(N)
                
                pc = np.zeros((len(kxs),2))
                for idx, (kx,ky) in enumerate(zip(kxs,kys)):
                    pc[idx,0] = ky[0] / dk + N/2
                    pc[idx,1] = kx[0] / dk + N/2

                self.add_point( pc, color = 'red')
                
    
    def add_point(self, locations, name = '', color = 'green'):
        radius = self.h.N // 30 # radius of the displayed circle 
        fullname = f'Carrier_{self.imageRaw_name}_{name}'
        try:
            self.viewer.layers[fullname].data = locations
        except:
            self.viewer.add_points(locations, size= radius,
                              face_color= [1,1,1,0], name = fullname , 
                              edge_width=0.5, edge_color=color) 
            
     
    def stack_demodulation(self): 
        hyperstack = self.get_imageRaw()
        if hyperstack.ndim == 5:
            angle_index = int(self.viewer.dims.current_step[-5]) 
            hyperstack = np.squeeze(hyperstack[angle_index,...]) 
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
        imageWFdata = np.squeeze(np.mean(self.get_imageRaw(), axis=-4))
        imname = 'WF_' + self.imageRaw_name
        scale = self.viewer.layers[self.imageRaw_name].scale 
        
        self.show_image(imageWFdata, imname, scale = np.delete(scale,-4))
        self.viewer.dims.current_step = [si//2 for si in imageWFdata.shape]


    @add_timer
    def calibration(self):  
        if hasattr(self, 'h'):
            data = self.get_imageRaw()
            dshape = data.shape
            zidx = int(self.viewer.dims.current_step[-3])
            delta = self.group.val // 2
            remainer = self.group.val % 2
            zmin = max(zidx-delta,0)
            zmax = min(zidx+delta+remainer,dshape[-3])
            new_delta = zmax-zmin
            data = data[...,zmin:zmax,:,:]
            
            phases_angles = self.phases_number.val*self.angles_number.val
            rdata = data.reshape(phases_angles, new_delta, dshape[-2],dshape[-1])            
            selected_imRaw = np.swapaxes(rdata, 0, 1).reshape((phases_angles * new_delta, dshape[-2],dshape[-1]))
            
            if self.gpu.val:
                self.h.calibrate_pytorch(selected_imRaw,self.find_carrier.val)
            else:
                self.h.calibrate(selected_imRaw,self.find_carrier.val)        
                
            self.isCalibrated = True
            #self.check_checkboxes()
            if self.find_carrier.val: # store the value found   
                self.kx_input = self.h.kx  
                self.ky_input = self.h.ky
                self.p_input = self.h.p
                self.ampl_input = self.h.ampl 
            self.show_carrier()
            self.show_xcorr()
             
            
    @add_timer
    def single_plane_reconstruction(self):  
        current_image = self.get_current_image()
        dshape= current_image.shape
        phases_angles = self.phases_number.val*self.angles_number.val
        rdata = current_image.reshape(phases_angles, dshape[-2],dshape[-1])
        
        if self.isCalibrated:
            print('recon executed')   
            if self.gpu.val:
                imageSIM = self.h.reconstruct_pytorch(rdata.astype(np.float32))
            else:
                imageSIM = self.h.reconstruct_rfftw(rdata)
            
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
        @add_timer_to_function
        def _stack_reconstruction():
            warnings.filterwarnings('ignore')
            stackSIM = np.zeros([sz,2*sy,2*sx], dtype=np.single)
            for zidx in range(sz):
                phases_stack = np.squeeze(pa_stack[:,zidx,:,:])
                if self.keep_calibrating.val:
                    delta = self.group.val // 2
                    if zidx % delta == 0: 
                        remainer = self.group.val % 2
                        zmin = max(zidx-delta,0)
                        zmax = min(zidx+delta+remainer,sz)
                        new_delta = zmax-zmin
                        data = pa_stack[:,zmin:zmax,:,:]
                        s_pa = data.shape[0]
                        selected_imRaw = np.swapaxes(data, 0, 1).reshape((s_pa * new_delta, sy, sx))
                        if self.gpu.val:
                            self.h.calibrate_pytorch(selected_imRaw,self.find_carrier.val)
                        else:
                            self.h.calibrate(selected_imRaw,self.find_carrier.val)                
                if self.gpu.val:
                    stackSIM[zidx,:,:] = self.h.reconstruct_pytorch(phases_stack.astype(np.float32))
                else:
                    stackSIM[zidx,:,:] = self.h.reconstruct_rfftw(phases_stack)
                    
            return stackSIM
        
        @thread_worker(connect={'returned': update_sim_image})
        @add_timer_to_function
        def _batch_reconstruction():
            warnings.filterwarnings('ignore')
            if self.gpu.val:
                stackSIM = self.h.batchreconstructcompact_pytorch(paz_stack, blocksize = 32)
            else:
                stackSIM = self.h.batchreconstructcompact(paz_stack)
            return stackSIM
        
        # main function exetuted here
        if not self.isCalibrated:
            raise(Warning('SIM processor not calibrated'))
        else:
            fullstack = self.get_imageRaw()
            dshape = fullstack.shape
            sz = dshape[-3]
            sy = dshape[-2]
            sx = dshape[-1]
            phases_angles = self.phases_number.val*self.angles_number.val
            pa_stack = fullstack.reshape(phases_angles, sz, sy, sx)
            paz_stack = np.swapaxes(pa_stack, 0, 1).reshape((phases_angles*sz, sy, sx))
            if self.batch.val:
                _batch_reconstruction()
            else: 
                _stack_reconstruction()
                
        
    def estimate_resolution(self, image:Image):
        from image_decorr import ImageDecorr
        @thread_worker
        def _estimate_resolution():
            warnings.filterwarnings('ignore')
            pixelsize = self.h.pixelsize / self.h.magnification
            dims = image.data.ndim
            if dims == 2:
                im = image.data
            elif dims ==3:
                z_index = int(self.viewer.dims.current_step[-3])
                im = image.data[z_index, :,:]
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
        phaseshift = np.zeros((7,3))
        expected_phase = np.zeros((7,3))
        error = np.zeros((7,3))
        
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i],self.h.ky[i],self.get_current_image())
            expected_phase[:,i] = np.arange(7) * 2*(i+1) * np.pi / 7
            phaseshift[:,i] = np.unwrap(phase - phase[0])
        error = phaseshift-expected_phase
        
        data_to_plot = [expected_phase,phaseshift, error]
        symbols = ['.','o','|']
        legend = ['expected', 'measured', 'error']
        self.plot_with_plt(data_to_plot, legend, symbols,
                                xlabel = 'step', ylabel = 'phase (rad)', vmax = 6*np.pi)
            
    def find_3phaseshifts(self):
        phase, _ = self.h.find_phase(self.h.kx,self.h.ky,self.get_current_image())
        expected_phase = np.arange(0, 2*np.pi ,2*np.pi / 3) 
        #phaseshift= np.unwrap(phase - expected_phase) - phase[0]
        phaseshift = np.unwrap(phase- phase[0])
        error = phaseshift-expected_phase
        data_to_plot = [expected_phase, phaseshift, error]
        symbols = ['.','o','|']
        legend = ['expected', 'measured', 'error']
        self.plot_with_plt(data_to_plot, legend, symbols,
                           xlabel = 'step', ylabel = 'phase (rad)', vmax = 2*np.pi)
        print(f"\nExpected phases: {expected_phase}\
                         \nMeasured phases: {phaseshift}\
                         \nError          : {error}\n")
                         
    
    def plot_with_plt(self, data_list, legend, symbols,
                      xlabel = 'step', ylabel = 'phase', vmax = 2*np.pi):
        import matplotlib.pyplot as plt
        char_size = 10
        plt.rc('font', family='calibri', size=char_size)
        fig = plt.figure(figsize=(4,3), dpi=150)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(xlabel, size=char_size)
        ax.set_ylabel(ylabel, size=char_size)
        
        s = data_list[0].shape
        cols = 1 if len(s)==1 else s[1]
        colors = ('black','red','green')
        for cidx in range(cols):
            color = colors[cidx%3]
            for idx, data in enumerate(data_list):
                column = data if len(s)==1 else data[...,cidx]
                marker = symbols[idx]
                linewidth = 0.2 if marker == 'o' else 0.8
                ax.plot(column, marker=marker, linewidth =linewidth, color=color)    
       
        ax.xaxis.set_tick_params(labelsize=char_size*0.75)
        ax.yaxis.set_tick_params(labelsize=char_size*0.75)
        ax.legend(legend, loc='best', frameon = False, fontsize=char_size*0.8)
        ax.grid(True, which='major', axis='both', alpha=0.2)
       
        vales_num = s[0]
        ticks = np.linspace(0, vmax*(vales_num-1)/vales_num, 2*vales_num-1 )
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
        
    def register_stack(self,image:Image, mode='Euclidean', plot_extent = 0.1):
    
        def add_image(data):
            self.viewer.add_image(data, 
                                scale = image.scale,
                                interpolation = 'bilinear',
                                name = f'registered_{image.name}')
            print('Registration completed')
        
        @thread_worker(connect={'returned': add_image})
        def _register_stack():
            warnings.filterwarnings('ignore')
            z_index = int(self.viewer.dims.current_step[-3])
            stack = image.data
            registered, wms = stack_registration(stack, z_idx=z_index, c_idx=0, method = 'cv2', mode=mode)
            
            nz = len(wms)
            dxs = np.zeros(nz)
            dys = np.zeros(nz)
            for widx,warp_matrix in enumerate(wms):
                dxs[widx] = warp_matrix[0,2]
                dys[widx] = warp_matrix[1,2]
            # self.plot_with_plt(dxs[:,np.newaxis], legend = [''], symbols=['o'],
            #                    xlabel ='frame z', ylabel ='displacement')
            # self.plot_with_plt(dys[:,np.newaxis], legend = [''], symbols=['o'],
            #                    xlabel ='frame z', ylabel ='displacement')
            zrange= np.arange(int(nz/2-nz*plot_extent),int(nz/2+nz*plot_extent), dtype=int)
            error = np.sum(np.sqrt((dxs[zrange]**2)+(dys[zrange]**2)))/np.size(zrange)
            print (f'average displacement: {error} pixels, frames: {np.amin(zrange)}-{np.amax(zrange)}')
            #print (f'rms displacement: {np.sqrt((np.sum(dys**2)+np.sum(dys**2))/nz)} pixels')
            
            return registered
            
        _register_stack() 


       
if __name__ == '__main__':
    file = 'test.tif'
    viewer = napari.Viewer()
    
    widget = HexSimAnalysis(viewer)
    mode={"choices": ['Translation','Affine','Euclidean','Homography']}
    registration = magicgui(widget.register_stack, call_button='Register stack', mode=mode)
    selection = magicgui(widget.select_layer, auto_call=True )# call_button='Select image layer')
    h5_opener = magicgui(widget.open_h5_dataset, call_button='Open h5 dataset')
    resolution = magicgui(widget.estimate_resolution, call_button='Estimate resolution')
    
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
    
    napari.run()      