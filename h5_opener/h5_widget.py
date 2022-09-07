# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:34:41 2022

@author: Andrea Bassi @Polimi
"""
from qtpy.QtWidgets import  QWidget
from napari.layers import Image
import numpy as np
from numpy.fft import fft2, fftshift 
from magicgui import magicgui
import pathlib
from get_h5_data import get_multiple_h5_datasets, get_h5_attr, get_datasets_index_by_name, get_group_name
import os

class H5opener(QWidget):
    
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()
        
    def open_h5_dataset(self, path: pathlib.Path = '',
                        dataset:int = 0, 
                        square = True,
                        cx = 1000, cy=600,size = 256):
        # open file
        directory, filename = os.path.split(path)
        t_idx = f'/t{dataset:04d}/'
        index_list, names = get_datasets_index_by_name(path, t_idx)
        stack,found = get_multiple_h5_datasets(path, index_list)
        
        if not square:
            stack = np.squeeze(stack)[:,np.newaxis,cy-size:cy+size,cx-size:cx+size]
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
        #sp,sz,sy,sx = stack.shape
        assert sy == sx, 'Non-square images are not supported'
        fullname = f'dts{dataset}_{filename}'
        self.show_image(stack, im_name=fullname)
        
    def _open_h5_dataset(self, path: pathlib.Path = '',
                        dataset:int = 0 ):
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
        #sp,sz,sy,sx = stack.shape
        assert sy == sx, 'Non-square images are not supported'
        fullname = f'dts{dataset}_{filename}'
        self.show_image(stack, im_name=fullname)
        self.rescaleZ()
        self.viewer.dims.axis_labels = ('phase','z','y','x')
                 
    def show_image(self, image_values, im_name, **kwargs):
        '''
        creates a new Image layer with image_values as data
        or updates an existing layer, if 'hold' in kwargs is True 
        '''
        if 'scale' in kwargs.keys():    
            scale = kwargs['scale']
        else:
            scale = [1.]*image_values.ndim
        if 'colormap' in kwargs.keys():
            colormap = kwargs['colormap']
        else:
            colormap = 'gray'    
        if kwargs.get('hold') is True and im_name in self.viewer.layers:
            layer = self.viewer.layers[im_name]
            layer.data = image_values
            layer.scale = scale
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = im_name,
                                            scale = scale,
                                            colormap = colormap,
                                            interpolation = 'bilinear')
        self.center_stack(image_values)
        self.move_layer_to_top(layer)
        if kwargs.get('autoscale') is True:
            layer.reset_contrast_limits()
        return layer

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
           

@magicgui(call_button = "Calculate Power Spectrum")
def calculate_spectrum(viewer: "napari.Viewer", image: Image,
                        log_scale: bool = True)->Image:
    stack = image.data
    #current_step = viewer.dims.current_step
    epsilon = 1e-6
    
    dims = stack.ndim
    if dims <2:
        print('no >2D data cannot calculate spectrum')
    
    if dims ==2:
        im= stack
    elif dims ==3:
        im = np.squeeze(stack[0,:,:])
    elif dims ==4:
        im = np.squeeze(stack[0,0,:,:])
    elif dims ==5:
        im = np.squeeze(stack[0,0,0,:,:])
  
    if log_scale:
        im_shown = np.log((np.abs(fftshift(fft2(im))))**2+epsilon)
        
    else:
        im_shown = np.abs(fftshift(fft2(im)))**2
    im_name = 'Spectrum_' + image.name
    return Image(im_shown, name = im_name)

  
if __name__ == '__main__':
    
    import napari
    MYPATH = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\HexSIM\\Frankenstein\\210726_163415_inv_bello_FLIR_NI_measurement.h5'
    viewer = napari.Viewer() 
    h5widget = H5opener(viewer)    
    h5_opener = magicgui(h5widget.open_h5_dataset, call_button='Open h5 dataset')
    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    
    from napari_sim_processor import SimAnalysis, reshape
    widget = SimAnalysis(viewer)
    my_reshape_widget = reshape()    
    viewer.window.add_dock_widget(my_reshape_widget, name = 'Reshape stack', add_vertical_stretch = True)
    viewer.window.add_dock_widget(widget,
                                  name = 'Sim analyzer @Polimi',
                                  add_vertical_stretch = True)
    napari.run()      