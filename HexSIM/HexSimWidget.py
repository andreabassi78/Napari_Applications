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
from widget_settings import Settings, add_timer
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
        
        self.phases_number = Settings('phases', dtype=int, initial=7, layout=slayout, 
                              write_function = self.reset_processor)
        self.angles_number = Settings('angles', dtype=int, initial=1, layout=slayout, 
                              write_function = self.reset_processor)
        self.magnification = Settings('M', dtype=float, initial=60, unit = 'X',  
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
        self.eta = Settings('eta', dtype=float, initial=0.65,
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
        self.showWiener = Settings('Show Wiener filter', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.show_wiener
                                     )
        self.showEta = Settings('Show Eta circle', dtype=bool, initial=False,
                                     layout=blayout,
                                     write_function = self.show_eta
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
        self.use_torch = Settings('Use Torch', dtype=bool, initial=False, layout=blayout, 
                           write_function = self.setReconstructor) 
        
        buttons_dict = {'Widefield': self.calculate_WF_image,
                        'Calibrate': self.calibration,
                        'Plot calibration phases':self.find_phaseshifts,
                        'SIM reconstruction': self.single_plane_reconstruction,
                        'Stack SIM reconstruction': self.stack_reconstruction,
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
        self.show_image(stack, fullname=fullname)
        self.rescaleZ()
        self.viewer.dims.axis_labels = ('phase','z','y','x')
            
        
    def select_layer(self, image: Image):
        '''
        Selects a Image layer assuming that it contains raw sim data organized
        as a stack, organized in one of the three following way:
            3D (phase,y,x)
            4D (phase,z,y,x)
            5D(angle,phase,z,y,x)'
        Changes the size of the Imagae layer to 5D (angle,phase,z,y,x), updated the labels and resets the Processor.
        Stores the name of the image in self.imageRaw_name, which is used frequently in the other methods.
        '''
        if not isinstance(image, Image):
            return
        if hasattr(self,'imageRaw_name'):
            delattr(self,'imageRaw_name')
        data = image.data
        if data.ndim == 5:
            pass
        elif data.ndim == 4:
            image.data = data[np.newaxis,...]
        elif data.ndim == 3:
            image.data = data[np.newaxis,:,np.newaxis,...]
        else:
            raise(KeyError('Please select a valid 3D(phase,y,x), 4D(phase,z,y,x) or 5D(angle,phase,z,y,x) stack'))
        self.imageRaw_name = image.name
        sa,sp,sz,sy,sx = image.data.shape
        assert sy == sx, 'Non-square images are not supported'
        self.angles_number.val = sa
        self.phases_number.val = sp
        self.viewer.dims.axis_labels = ["angle", "phase", "z", "y","x"]
        self.rescaleZ()
        self.center_stack(image)
        self.start_sim_processor()
        self.move_layer_to_top(image)
        self.showXcorr.val = False
        self.showSpectrum.val = False
        self.showEta.val = False
        self.showCarrier.val = False
        self.keep_calibrating.val = False
        self.keep_reconstructing.val = False
        print(f'Selected image layer: {image.name}')
           
            
    def rescaleZ(self):
        '''
        changes the z-scale of all the Images in layer with shape >=3D
        '''
        self.zscaling = self.dz.val /(self.pixelsize.val/self.magnification.val)
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                if layer.ndim >=3:
                    scale = layer.scale 
                    scale[-3] = self.zscaling
                    layer.scale = scale
      
    def center_stack(self, image_layer):
        '''
        centers a >3D stack in z,y,x 
        '''
        data = image_layer.data
        if data.ndim >2:
            current_step = list(self.viewer.dims.current_step)
            for dim_idx in [-3,-2,-1]:
                current_step[dim_idx] = data.shape[dim_idx]//2
            self.viewer.dims.current_step = current_step                
    
           
    def on_step_change(self, *args):   
        if hasattr(self, 'imageRaw_name'): #self.viewer.dims.ndim >3:
            self.setReconstructor()
            if self.showSpectrum.val:
                 self.show_spectrum()
            
            
    def show_image(self, image_values, fullname, **kwargs):
        '''
        creates a new Image layer or updates an existing one if 'hold' in kwargs is True 
        '''
        if 'scale' in kwargs.keys():    
            scale = kwargs['scale']
        else:
            scale = [1.]*image_values.ndim
        if 'colormap' in kwargs.keys():
            colormap = kwargs['colormap']
        else:
            colormap = 'gray'    
        if kwargs.get('hold') is True and fullname in self.viewer.layers:
            layer = self.viewer.layers[fullname]
            layer.data = image_values
            layer.scale = scale
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = fullname,
                                            scale = scale,
                                            colormap = colormap,
                                            interpolation = 'bilinear')
        self.center_stack(image_values)
        self.move_layer_to_top(layer)
        if kwargs.get('autoscale') is True:
            layer.reset_contrast_limits()
        return layer


    def remove_layer(self, layer):
        if layer in self.viewer.layers:
            self.viewer.layers.remove(layer.name)    
    
    
    def move_layer_to_top(self, layer_to_move):
        '''
        Moves the layer to the top of the viewer and selects it
        '''
        for idx,layer in enumerate(self.viewer.layers):
            #couldn't find a way to get the index of a certain layer directly from the Layer object
            if layer is layer_to_move:
                self.viewer.layers.move(idx, len(self.viewer.layers))
                layer.visible = True
                if isinstance(layer, Image):
                    viewer.layers.selection = [layer]
    
    
    def make_layers_visible(self, *layers_list):
        '''
        Makes all the passed layers visible, while making all the others invisible
        '''
        for layer in self.viewer.layers:
            if layer in layers_list:
                layer.visible = True
            else:
                layer.visible = False    
    
    
    def is_image_in_layers(self):
        '''
        Checks if the raw image has been selected
        '''
        if hasattr(self, 'imageRaw_name'):
            if self.imageRaw_name in self.viewer.layers:
                return True
        return False   
    
    
    def get_hyperstack(self):
        '''
        Returns the full 5D raw image stack
        '''
        try:
            return self.viewer.layers[self.imageRaw_name].data
        except:
             raise(KeyError('Please select a valid stack'))
    
    
    def get_current_stack(self):
        '''
        Returns the raw image stack at the z value selected in the viewer  
        '''
        fullstack = self.get_hyperstack()
        z_index = int(self.viewer.dims.current_step[2])
        s = fullstack.shape
        assert z_index < s[-3], 'Please choose a valid z step for the selected stack'
        stack = fullstack[:,:,z_index,:,:]
        return stack
    
    
    def get_current_image(self):
        '''
        Returns the raw image stack at the z, angle and phase values selected in the viewer  
        '''
        hs = self.get_hyperstack()
        z_index = int(self.viewer.dims.current_step[2])
        phase_index = int(self.viewer.dims.current_step[1])
        angle_index = int(self.viewer.dims.current_step[0])   
        img0 = hs[angle_index,phase_index,z_index,:,:]
        return(img0)
    
    
    def start_sim_processor(self):
        ''''
        Creates an instance of the Processor
        '''
        self.isCalibrated = False
        if hasattr(self, 'h'):
            self.stop_sim_processor()
            self.start_sim_processor()
        else:
            if self.phases_number.val == 3 and self.angles_number.val == 1: 
                self.h = SimProcessor()  
                k_shape = (1,1)
            elif self.phases_number.val == 7:  
                self.h = HexSimProcessor()  
                k_shape = (3,1)
            elif self.phases_number.val == 3 and self.angles_number.val == 3: 
                self.h = ConvSimProcessor()
                k_shape = (3,1)
            else: 
                raise(ValueError("Invalid phases or angles number"))
            self.h.debug = False
            self.setReconstructor() 
            self.kx_input = np.zeros(k_shape, dtype=np.single)
            self.ky_input = np.zeros(k_shape, dtype=np.single)
            self.p_input = np.zeros(k_shape, dtype=np.single)
            self.ampl_input = np.zeros(k_shape, dtype=np.single)

            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')
  
    
    def reset_processor(self,*args):
        self.isCalibrated = False
        self.stop_sim_processor()
        self.start_sim_processor()
       
        
    @add_timer    
    def setReconstructor(self,*args):
        '''
        Sets the attributes of the Processor
        Executed frequently, upon update of several Settings
        '''
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
            if self.showEta.val:
                self.show_eta()
          
            
    def show_wiener(self, *args):
        """
        Shows the Wiener filter 
        """
        if self.is_image_in_layers():
            imname = 'Wiener_' + self.imageRaw_name
            if self.isCalibrated and self.showWiener.val:
                
                img = self.h.wienerfilter
                swy,swx = img.shape
                self.show_image(img[swy//2-swy//4:swy//2+swy//4,swx//2-swx//4:swx//2+swx//4],
                                imname, hold = True, scale=[1,1])
                self.show_carrier()
                self.show_eta()
            elif not self.showWiener.val and imname in self.viewer.layers:
                self.remove_layer(self.viewer.layers[imname])
                       
            
    def show_spectrum(self, *args):
        """
        Calculates power spectrum of the image
        """
        if self.is_image_in_layers():
            imname = 'Spectrum_' + self.imageRaw_name
            if self.showSpectrum.val:
                img0 = self.get_current_image()
                epsilon = 1e-10
                ps = np.log((np.abs(fftshift(fft2(img0))))**2+epsilon)
                self.show_image(ps, imname, hold = True)
                self.show_carrier()
                self.show_eta()
            elif not self.showSpectrum.val and imname in self.viewer.layers:
                self.remove_layer(self.viewer.layers[imname])
       
     
    def show_xcorr(self, *args):
        """
        Show the crosscorrelation of the low and high pass filtered version of the raw images,
        used forfinding the carrier
        """
        if self.is_image_in_layers():
            imname = 'Xcorr_' + self.imageRaw_name
            if self.showXcorr.val and self.isCalibrated:
                ixf = self.h.ixf
                self.show_image(ixf, imname, hold = True,
                                colormap ='twilight', autoscale = True)
                self.show_carrier()
                self.show_eta()
            elif not self.showXcorr.val and imname in self.viewer.layers:
                self.remove_layer(self.viewer.layers[imname])
            
     
    def calculate_kr(self,N):  
        '''
        Parameter:
            N: number of pixels of the image
        Returns: 
            cutoff: pupil cutoff frequency in pixels number
            dk: sampling in spatial frequancy domain
        '''
        dx = self.h.pixelsize / self.h.magnification  # Sampling in image plane
        res = self.h.wavelength / (2 * self.h.NA)
        cutoff = 1/res/2 # coherent cutoff frequency
        oversampling = res / dx
        dk = oversampling / (N / 2)  
        cutoff_in_pixels = cutoff / dk
        return cutoff_in_pixels, dk   
      
      
    def show_carrier(self, *args):
        '''
        Draw carrier frenquencies in a shape layer
        '''
        if self.is_image_in_layers() and self.isCalibrated:
            # shows carrier frequencies
            name = f'carrier_{self.imageRaw_name}'
            if self.showCarrier.val:
                N = self.h.N
                cutoff, dk = self.calculate_kr(N)
                kxs = self.h.kx
                kys = self.h.ky
                pc = np.zeros((len(kxs),2))
                for idx, (kx,ky) in enumerate(zip(kxs,kys)):
                    pc[idx,0] = ky[0] / dk + N/2
                    pc[idx,1] = kx[0] / dk + N/2
                radius = self.h.N // 30 # radius of the displayed circle 
                self.add_circles(pc, radius, name, color='red')
                # kr = np.sqrt(kxs**2+kys**2)
                # print('Carrier magnitude / cut off:', *kr/cutoff*dk)
            elif name in self.viewer.layers:
                self.remove_layer(self.viewer.layers[name])
                
           
    def show_eta(self, *args):
        '''
        Shows two circles with radius eta (green circle), 
        and with the radius of the pupil (blue) 
        '''
        if self.is_image_in_layers():
            name = f'eta_circle_{self.imageRaw_name}'
            if self.showEta.val:
                N = self.h.N
                cutoff, dk   = self.calculate_kr(N)  
                eta_radius = self.h.eta * cutoff
                self.add_circles(np.array([N/2,N/2]), eta_radius,
                               name, color='green')
                self.add_circles(np.array([N/2,N/2]), cutoff,
                               name, color='blue', hold=True)
            elif name in self.viewer.layers:
                self.remove_layer(self.viewer.layers[name])


    def add_circles(self, locations, radius=20,
                    shape_name='shapename', color='blue', hold=False
                    ):
        '''
        locations : np.array with yx cohordinates of the center 
        shape_name : str, name of the new Shape
        radius : radius of the circles
        color : str of RGBA list, color of the circles
        hold : bool, if Trueu pdates the existing layer, with name shape_name,
            without creating a new one
        '''
        ellipses = []
        for center in locations: 
            bbox = np.array([center+np.array([radius, radius]),
                             center+np.array([radius,-radius]),
                             center+np.array([-radius,-radius]),
                             center+np.array([-radius, radius])]
                            )
            ellipses.append(bbox)
        
        if shape_name in self.viewer.layers: 
            circles_layer = self.viewer.layers[shape_name]
            if hold:
                circles_layer.add_ellipses(ellipses, edge_color=color)
            else:
                circles_layer.data = np.array(ellipses)
        else:  
            circles_layer = self.viewer.add_shapes(name=shape_name,
                                   edge_width = 0.8,
                                   face_color = [1,1,1,0],
                                   edge_color = color)
            circles_layer.add_ellipses(ellipses, edge_color=color)
        self.move_layer_to_top(circles_layer)   
    
    
    def showCalibrationTable(self):
        import pandas as pd
        headers= ['kx_in','ky_in','kx','ky','phase','amplitude']
        vals = [self.kx_input, self.ky_input,
                self.h.kx, self.h.ky,
                self.h.p,self.h.ampl]
        table = pd.DataFrame([vals] , columns = headers )
        print(table)
            
            
    def _stack_demodulation(self): 
        hyperstack = self.get_hyperstack()
        angle_index = int(self.viewer.dims.current_step[0]) 
        hyperstack = np.squeeze(hyperstack[angle_index,...]) 
        p,z,y,x = hyperstack.shape
        demodulated = np.zeros([z,y,x]).astype('complex64')
        for frame_index in range(z): 
            for p_idx in range(p):
                demodulated[frame_index,:,:] += 2/p * hyperstack[p_idx,frame_index,:,:]*np.exp(1j*2*np.pi*p_idx/p)
        demodulated_abs = np.abs(demodulated).astype('float') 
        imname = 'Demodulated_' + self.imageRaw_name
        scale = [self.zscaling,1,1]
        self.show_image(demodulated_abs, imname, scale= scale, hold = True)
        print('Stack demodulation completed')
        

    def stack_demodulation(self):
        '''
        Demodulates the data as proposed in Neil et al, Optics Letters 1997.
        '''
        if not self.isCalibrated:
            raise(Warning('SIM processor not calibrated'))
        fullstack = self.get_hyperstack()
        sa,sp,sz,sy,sx = fullstack.shape
        phases_angles = sa*sp
        pa_stack = fullstack.reshape(phases_angles, sz, sy, sx)
        demodulated = np.zeros([sz,sy,sx]).astype('float')
        # if self.use_torch.val: TODO implement demodulation in torch, add to function call.astype(np.float32)
        #     demodulation_function = self.h.OSreconstruct_pytorch
        demodulation_function = self.h.filteredOSreconstruct
        for frame_index in range(sz):
            stack = np.squeeze(pa_stack[:,frame_index,:,:])
            demodulated[frame_index,:,:] = np.squeeze(demodulation_function(stack))
        imname = 'Demodulated_' + self.imageRaw_name
        scale = [self.zscaling,1,1]
        self.show_image(demodulated, imname, scale= scale, hold = True)
        #print('Stack demodulation completed')
        
    
    def calculate_WF_image(self):
        '''
        Calculates and shows the widefield image from the raw 5D image stack.
        It averages the data on all phases and angles.
        Shows the resulting 3D image stack as an Image layer of the viewer.
        '''
        imageWFdata = np.mean(self.get_hyperstack(), axis=(0,1))
        imname = 'WF_' + self.imageRaw_name
        scale = self.viewer.layers[self.imageRaw_name].scale
        self.show_image(imageWFdata, imname, scale = scale[2:], hold = True, autoscale = True)
        

    def calibration(self):
        '''
        Performs the data calibration using the Processor (self.h).
        It is performed on a stack of images around the frame selected in the viewer.
        The size of the stack is the value specified in the "group" Setting.
        '''
        if hasattr(self, 'h'):
            data = self.get_hyperstack()
            dshape = data.shape
            zidx = int(self.viewer.dims.current_step[2])
            delta = self.group.val // 2
            remainer = self.group.val % 2
            zmin = max(zidx-delta,0)
            zmax = min(zidx+delta+remainer,dshape[2])
            new_delta = zmax-zmin
            data = data[...,zmin:zmax,:,:]
            phases_angles = self.phases_number.val*self.angles_number.val
            rdata = data.reshape(phases_angles, new_delta, dshape[-2],dshape[-1])            
            selected_imRaw = np.swapaxes(rdata, 0, 1).reshape((phases_angles * new_delta, dshape[-2],dshape[-1]))
            if self.use_torch.val:
                self.h.calibrate_pytorch(selected_imRaw,self.find_carrier.val)
            else:
                self.h.calibrate(selected_imRaw,self.find_carrier.val)
            self.isCalibrated = True
            if self.find_carrier.val: # store the value found   
                self.kx_input = self.h.kx  
                self.ky_input = self.h.ky
                self.p_input = self.h.p
                self.ampl_input = self.h.ampl 
            self.show_wiener()
            self.show_xcorr()
            self.show_carrier()
            self.show_eta()
            
                   
    def single_plane_reconstruction(self):
        '''
        Performs SIM reconstruction on the selected z plane.
        '''
        current_image = self.get_current_stack()
        dshape= current_image.shape
        phases_angles = self.phases_number.val*self.angles_number.val
        rdata = current_image.reshape(phases_angles, dshape[-2],dshape[-1])
        if self.isCalibrated:
            if self.use_torch.val:
                imageSIM = self.h.reconstruct_pytorch(rdata.astype(np.float32)) #TODO:this is left after conversion from torch
            else:
                imageSIM = self.h.reconstruct_rfftw(rdata)
            imname = 'SIM_' + self.imageRaw_name
            self.show_image(imageSIM, fullname=imname, scale=[0.5,0.5], hold =True, autoscale = True)
        else:
            raise(Warning('SIM processor not calibrated'))  
    
    
    def stack_reconstruction(self):
        '''
        Performs SIM reconstruction on entire data (5D raw image stack).
        Performs plane-by-plane reconstruction (_stack_reconstruction), 
            calibrating the Processor continuosly if "Continuous calibration" checkbox is selected
        Performs batch reconstruction if "Batch reconstrction" checkbox is selected
        '''
        def update_sim_image(stack):
            imname = 'SIMstack_' + self.imageRaw_name
            scale = [self.zscaling, 0.5, 0.5]
            self.show_image(stack, fullname=imname, scale=scale, hold = True, autoscale = True)
            
            print('Stack reconstruction completed')
        
        @thread_worker(connect={'returned': update_sim_image})
        @add_timer
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
                        if self.use_torch.val:
                            self.h.calibrate_pytorch(selected_imRaw,self.find_carrier.val)
                        else:
                            self.h.calibrate(selected_imRaw,self.find_carrier.val)                
                if self.use_torch.val:
                    stackSIM[zidx,:,:] = self.h.reconstruct_pytorch(phases_stack.astype(np.float32)) #TODO:this is left after conversion from torch
                else:
                    stackSIM[zidx,:,:] = self.h.reconstruct_rfftw(phases_stack)      
            return stackSIM
        
        @thread_worker(connect={'returned': update_sim_image})
        @add_timer
        def _batch_reconstruction():
            warnings.filterwarnings('ignore')
            if self.use_torch.val:
                stackSIM = self.h.batchreconstructcompact_pytorch(paz_stack, blocksize = 32)
            else:
                stackSIM = self.h.batchreconstructcompact(paz_stack)
            return stackSIM
        
        # main function exetuted here
        if not self.isCalibrated:
            raise(Warning('SIM processor not calibrated'))
        fullstack = self.get_hyperstack()
        sa,sp,sz,sy,sx = fullstack.shape
        phases_angles = sa*sp
        pa_stack = fullstack.reshape(phases_angles, sz, sy, sx)
        paz_stack = np.swapaxes(pa_stack, 0, 1).reshape((phases_angles*sz, sy, sx))
        if self.batch.val:
            _batch_reconstruction()
        else: 
            _stack_reconstruction()
                
          
    def find_phaseshifts(self):
        if self.isCalibrated:
            if self.phases_number.val==7:
                self.find_hexsim_phaseshifts()
            elif self.phases_number.val==3 :
                self.find_sim_phaseshifts()
            #self.showCalibrationTable() 
        else:
            raise(Warning('SIM processor not calibrated, unable to show phases'))
            
        
    def find_hexsim_phaseshifts(self):   
        phaseshift = np.zeros((7,3))
        expected_phase = np.zeros((7,3))
        error = np.zeros((7,3))
        stack = self.get_current_stack()
        sa,sp,sy,sx = stack.shape
        img = stack.reshape(sa*sp, sy, sx) 
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i], self.h.ky[i], img)
            expected_phase[:,i] = np.arange(7) * 2*(i+1) * np.pi / 7
            phaseshift[:,i] = np.unwrap(phase - phase[0])
        error = phaseshift-expected_phase
        data_to_plot = [expected_phase, phaseshift, error]
        symbols = ['.','o','|']
        legend = ['expected', 'measured', 'error']
        self.plot_with_plt(data_to_plot, legend, symbols,
                                xlabel = 'step', ylabel = 'phase (rad)', vmax = 6*np.pi)
            
    
    def find_sim_phaseshifts(self):   
        stack = self.get_current_stack()
        sa,sp,sy,sx = stack.shape
        img = stack.reshape(sa*sp, sy, sx)  
        for angle_idx in range (sa):
            phaseshift = np.zeros((sp,sa))
            expected_phase = np.zeros((sp,sa))
            error = np.zeros((sp,sa))
            phase, _ = self.h.find_phase(self.h.kx[angle_idx], self.h.ky[angle_idx], img)
            phase = np.unwrap(phase)
            phase = phase.reshape(sa,sp).T
            expected_phase[:,angle_idx] = np.arange(sp) * 2*np.pi / sp
            phaseshift= phase-phase[0,:]
            error = phaseshift-expected_phase      
            data_to_plot = [expected_phase, phaseshift, error]
            symbols = ['.','o','|']
            legend = ['expected', 'measured', 'error']
            self.plot_with_plt(data_to_plot, legend, symbols, title = f'angle {angle_idx}',
                                    xlabel = 'step', ylabel = 'phase (rad)', vmax = 2*np.pi)
                             
    
    def plot_with_plt(self, data_list, legend, symbols,
                      xlabel = 'step', ylabel = 'phase',
                      vmax = 2*np.pi, title = ''):
        import matplotlib.pyplot as plt
        char_size = 10
        plt.rc('font', family='calibri', size=char_size)
        fig = plt.figure(figsize=(4,3), dpi=150)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(xlabel, size=char_size)
        ax.set_ylabel(ylabel, size=char_size)
        ax.set_title(title, size=char_size)
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


    def estimate_resolution(self, image:Image):
        from image_decorr import ImageDecorr
        @thread_worker
        def _estimate_resolution():
            warnings.filterwarnings('ignore')
            pixelsize = self.h.pixelsize / self.h.magnification
            dims = image.data.ndim
            if dims==2:
                im = image.data
            elif dims==3:
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

  
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    
    widget = HexSimAnalysis(viewer)
    
    mode={"choices": ['Translation','Affine','Euclidean','Homography']}
    registration = magicgui(widget.register_stack, call_button='Register stack', mode=mode)
    selection = magicgui(widget.select_layer, call_button='Select image layer')
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