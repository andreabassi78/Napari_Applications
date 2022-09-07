# -*- coding: utf-8 -*-
"""
Created on Sep 7 23:34:41 2022

@author: Andrea Bassi @Polimi
"""

import napari
from qtpy.QtWidgets import QWidget
from magicgui import magicgui
import pathlib

class Opener(QWidget):
  
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()
            
    def open_h5_dataset(self, path: pathlib.Path = '',
                        dataset:int = 0):
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
        #sp,sz,sy,sx = stack.shape
        assert sy == sx, 'Non-square images are not supported'
        fullname = f'dts{dataset}_{filename}'
        self.viewer.add_image(stack, name=fullname)
        self.viewer.dims.axis_labels = ('phase','z','y','x')
  
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    
    widget = Opener(viewer)
    
    h5_opener = magicgui(widget.open_h5_dataset, call_button='Open h5 dataset')
    
    
    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
   
    napari.run()      