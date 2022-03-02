# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:09:58 2022

@author: andrea
"""
import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image,Labels, Shapes
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_local, threshold_mean
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import erosion, dilation, closing, opening, cube, ball, remove_small_objects, remove_small_holes
from qtpy.QtWidgets import QLabel, QVBoxLayout,QSplitter, QHBoxLayout, QWidget, QPushButton, QSpinBox, QFormLayout
from napari.qt.threading import thread_worker


class Segmentation3D():
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        
                                 
    def make_layers_visible(self, *layers_list):
        
        for layer in self.viewer.layers:
            if layer in layers_list or type(layer) is Shapes:
                layer.visible = True
            else:
                layer.visible = False
    
    def show_props(self, labels_layer:Labels):
        from skimage.measure import regionprops_table
        import pandas as pd
        if labels_layer.ndim==4:
            labels_stack = np.squeeze(labels_layer.data)
        else:
            labels_stack = labels_layer.data
        properties = ['label', 'area',# 'solidity', 'extent',
                      'major_axis_length', 'minor_axis_length',
                      'centroid']

        table = pd.DataFrame(regionprops_table(labels_stack, properties=properties))
        pd.set_option('max_columns', table.columns.size)
        print(table)


    def rescale(self,
                image_layer:Image,
                dx:float = 1,
                dy:float = 1,
                dz:float = 1,
                scalebar:bool=False,
                ):
        
        
        if image_layer.ndim == 4:
            image_layer.scale = [1,dz,dy,dx]
        
        elif image_layer.ndim ==3:
            image_layer.scale = [dz,dy,dx]
        
        elif image_layer.ndim ==2:
            image_layer.scale = [dy,dx]
        
        else:
             raise(ValueError('image layer dimensionality not supported'))

        self.viewer.scale_bar.visible = scalebar
        self.viewer.layers.selection = [image_layer]
    
    def segment(self,
                image_layer:Image,
                find_thresold:bool = False,
                use_current_step:bool = False,
                method:str = 'otsu',
                thresold:int = 0,
                block_size:int = 20,
                median_filter_size:int = 1,
                erosion_size:int = 2,
                remove_smaller_than:int = 1,
                fill_holes:bool = False,
                smooth_size:int = 2,
                show_labels:bool = True                
                ):
   
        def add_labels(labels_stack):
            
            label_name= 'segmented_' + image_layer.name
            
            if image_layer.ndim == 4:
                data = labels_stack[np.newaxis,...] 
            
            elif image_layer.ndim in (2,3):
                data = labels_stack
            
            try: 
                 self.viewer.layers[label_name].data = data 
                
            except:
                self.viewer.add_labels(data,
                                        name=label_name,
                                        scale = image_layer.scale
                                        )
            self.make_layers_visible(self.viewer.layers[label_name], image_layer)
            self.viewer.layers.selection = [image_layer]
            print(f'Segmentation completed, found {np.amax(data)} labels')
            
        
        @thread_worker(connect={'returned': add_labels})
        def _segment():
            import warnings
            warnings.filterwarnings('ignore')
        
            if image_layer.ndim == 4:
                
                #current_step = self.viewer.dims.current_step
                data= np.squeeze(np.array(image_layer.data))
            
            elif image_layer.ndim ==3:
            # #     #current_step = self.viewer.dims.current_step
                data= np.array(image_layer.data)
            
            else:
                 raise(ValueError('image layer dimensionality not supported'))
                
            if median_filter_size > 0:
                data = ndi.median_filter(data, size=median_filter_size)
            
            if find_thresold and method != 'local':
                 
                thres_dict = {'otsu': threshold_otsu, 
                              'li':threshold_li,
                              'yen':threshold_yen,
                              'local':threshold_local,
                              'mean':threshold_mean }
                 
                threshold_func = thres_dict[method]
                
                if use_current_step :
                    z = viewer.dims.current_step[-3]
                    _data = data[...,z,:,:]
                else:
                    _data = data
                
                thres = threshold_func(_data)
                segment_widget.thresold.value = thres                 
         
            
           
            elif find_thresold and method == 'local':
                thres = np.zeros_like(data)
                bs = block_size # = _data.shape[-1]//20
                if bs % 2 == 0:
                    bs=block_size+1
                for z in range(data.shape[-3]):    
                    _data = data[...,z,:,:]
                    thres_z = threshold_local(_data, bs)
                    thres[...,z,:,:] = thres_z
                segment_widget.thresold.value = np.amax(thres)  
                     
            else:
                thres = thresold
            
            bw = data > thres            
            
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
            
            return labels_stack
        _segment()
        
        
                
    
if __name__ == '__main__':
   
    viewer = napari.Viewer()
    myclass = Segmentation3D(viewer)
  
    show_props_widget = magicgui(myclass.show_props, call_button='Print region properties')
    segment_widget = magicgui(myclass.segment,
                              call_button='Run segmentation',
                              method={"choices": ['otsu', 'yen', 'li', 'local', 'mean']},)
    segment_widget.thresold.max = 2**16
    rescale_widget =magicgui(myclass.rescale, call_button='Set scale')
    
    viewer.window.add_dock_widget(segment_widget,
                                  name = '3D segmentation @Polimi',
                                  add_vertical_stretch = True)
    viewer.window.add_dock_widget(rescale_widget,
                                  name='Rescale',
                                  add_vertical_stretch = True)
    viewer.window.add_dock_widget(show_props_widget,
                                   name = 'Region prop',
                                   add_vertical_stretch = True)
    
    try:
        #path = "C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\STED_Humanitas\\sample_data\\C0421_Pre-Post_STED_18621_OK.lif - C0421_GphN_vGAT_CTRL_N1.tif"
        path = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\DatasetTestNapari\\registered_ds_30.tif'
        viewer.open(path)
        #viewer.dims.axis_labels = ('c','z','y','x')
    finally:

    
        napari.run() 