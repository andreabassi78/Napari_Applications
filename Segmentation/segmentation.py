# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:09:58 2022

@author: andrea
"""
import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image,Labels
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_local, threshold_mean
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import erosion, dilation, closing, opening, cube, ball, remove_small_objects, remove_small_holes
from qtpy.QtWidgets import QLabel, QVBoxLayout,QSplitter, QHBoxLayout, QWidget, QPushButton, QSpinBox, QFormLayout


class Segmentation3D():
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
                                 
    def make_layers_visible(self, *layers_list):
        
        for layer in self.viewer.layers:
            if layer in layers_list:
                layer.visible = True
            else:
                layer.visible = False
    
    def show_props(self, labels_layer:Labels):
        from skimage.measure import regionprops_table
        import pandas as pd
        labels_stack = np.squeeze(labels_layer.data)
        properties = ['label', 'area', 'solidity', 'extent',
                      'major_axis_length', 'minor_axis_length',
                      'centroid']

        table = pd.DataFrame(regionprops_table(labels_stack, properties=properties))
        pd.set_option('max_columns', table.columns.size)
        print(table)

    
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
   
        if image_layer.ndim == 4:
            
            current_step = self.viewer.dims.current_step
        
            data= np.squeeze(np.array(image_layer.data))
            
            if median_filter_size > 0:
                data = ndi.median_filter(data, size=median_filter_size)
            
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
            
            label_name= 'segmented_' + image_layer.name
            
            try: 
                 self.viewer.layers[label_name].data = labels_stack[np.newaxis,...] 
                
            except:
                self.viewer.add_labels(labels_stack[np.newaxis,...],
                                        name=label_name,
                                        scale = image_layer.scale
                                        )
            self.make_layers_visible(self.viewer.layers[label_name], image_layer)
            self.viewer.layers.selection = [image_layer]
            self.viewer.scale_bar.visible = True
                
    
if __name__ == '__main__':
   
    viewer = napari.Viewer()
    myclass = Segmentation3D(viewer)
  
    show_props_widget = magicgui(myclass.show_props, call_button='Print region properties')
    segment_widget = magicgui(myclass.segment, call_button='Run segmentation')
    viewer.window.add_dock_widget(segment_widget,
                                  name = '3D segmentation @Polimi',
                                  add_vertical_stretch = True)
    viewer.window.add_dock_widget(show_props_widget,
                                  name = 'Region prop',
                                  add_vertical_stretch = True)
    
    path = "C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\STED_Humanitas\\sample_data\\C0421_Pre-Post_STED_18621_OK.lif - C0421_GphN_vGAT_CTRL_N1.tif"
    viewer.open(path)
    
    viewer.dims.axis_labels = ('c','z','y','x')
    
    napari.run() 