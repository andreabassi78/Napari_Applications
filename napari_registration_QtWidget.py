# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:09:58 2022

@author: andrea
"""
import napari
from registration_utils import plot_data, save_in_excel, normalize_stack, select_rois_from_stack
from registration_utils import align_with_registration, update_position
from registration_utils import calculate_spectrum, correct_decay 
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
from skimage.measure import regionprops
from napari.layers import Image, Points, Labels,Shapes
import pathlib
import os
import time
from napari.qt.threading import thread_worker
import pandas as pd
from typing import Union
import numpy as np
import dask.array as da


class RegistrationWidget(QWidget):
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        
        # initialize layout
        layout = QGridLayout()
        
        # add subtract background button
        subtract_btn = QPushButton('Subtract background', self)
        subtract_btn.clicked.connect(self.subtract_background)
        layout.addWidget(subtract_btn)
        
        # add registration button
        registration_btn = QPushButton('Register ROIs', self)
        registration_btn.clicked.connect(self.register_rois)
        layout.addWidget(registration_btn)
        
        # add processing button
        process_btn = QPushButton('Process ROIs', self)
        process_btn.clicked.connect(self.process_rois)
        layout.addWidget(process_btn)
        
        # add normalize checkbox
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setChecked(False)
        layout.addWidget(self.normalize_checkbox)
        
        # add plot data checkbox
        self.plot_checkbox = QCheckBox("Plot")
        self.plot_checkbox.setChecked(True)
        layout.addWidget(self.plot_checkbox)
        
        # add save checkbox
        self.save_checkbox = QCheckBox("Save")
        self.save_checkbox.setChecked(False)
        layout.addWidget(self.save_checkbox)
        
        # print(self.viewer.layers.selection) # shows the currently selected layes
        
        # activate layout
        self.setLayout(layout) # QWidget method
        
    
    def update_image(self):
        try:
            # if the layer exists, update the data
            self.viewer.layers['corrected'].data = self.corrected
        except KeyError:
            # otherwise add it to the viewer
            self.viewer.add_image(self.corrected, name='corrected')
    
    
    @thread_worker(connect={'yielded': update_image})
    def subtract_background(self,*args):
        image_layer = self.viewer.layers['images'] # TODO learn how to link to gui menu
        labels_layer = self.viewer.layers['labels']
        AMAX=2**16-1
        #check_if_suitable_layer(image, labels, roitype='registration')
        original = np.asarray(image_layer.data)
        self.corrected = np.zeros_like(original) 
        mask = np.asarray(labels_layer.data[0,...]>0).astype(int)
        
        for plane_idx, plane in enumerate(original.astype('int')):
            print(f'Correcting background on frame {plane_idx}')
            props = regionprops(label_image=mask,
                                intensity_image=plane)
            intensities = np.zeros(len(props))
            for roi_idx, prop in enumerate(props):
                intensities[roi_idx]=prop['mean_intensity']
            background = np.mean(intensities)
            diff = np.clip(plane-background, a_min=0, a_max=AMAX).astype('uint16')
            self.corrected[plane_idx,:,:] = diff
            yield self
            
    
        
    def register_rois(self, roi_size: int = 100,  
                median_filter_size:int = 3, 
                time_series_undersampling:int =1 # TODO remove after testing
        ):
        
                
        image_layer = self.viewer.layers['images']
        initial_rois_layer = self.viewer.layers['labels']
        check_if_suitable_layer(image_layer,initial_rois_layer, roitype='registration')

        stack = image_layer.data
        
        print(initial_rois_layer.data.shape)

        props = regionprops(label_image=np.asarray(initial_rois_layer.data).astype(int),
                                intensity_image=np.asarray(stack))
        initial_positions = []
        for prop in props:
            initial_positions.append(prop['centroid'])
        
        normalize = True 
        if normalize: # this option is obsolete
            stack, _vmin, _vmax = normalize_stack(stack)
        
        previous_rois = select_rois_from_stack(stack, initial_positions, roi_size)
        registered_positions = []
        next_positions = initial_positions.copy()
        registered_positions.append(next_positions)
        
        roi_num = len(initial_positions)
        time_frames_num, sy, sx = stack.shape
        
        for t_index in range(1, time_frames_num, time_series_undersampling):

            next_rois = select_rois_from_stack(stack, next_positions, roi_size)
            # registration based on opencv function
            aligned, original, dx, dy = align_with_registration(next_rois,
                                                                previous_rois,
                                                                median_filter_size,
                                                                roi_size
                                                                )
            next_positions = update_position(next_positions,
                                             dz = time_series_undersampling,
                                             dx_list = dx,
                                             dy_list = dy
                                             )
            registered_positions.append(next_positions)
        
        # positions must be reshaped in a single list, to be converted in napari points 
        centers = np.reshape(np.array(registered_positions),
                                        [time_frames_num*roi_num, stack.ndim])
        print(f'Registered {time_frames_num//time_series_undersampling} frames. ')
        self.viewer.add_points(centers,size=roi_size, edge_width=roi_size//25+1, opacity = 0.3,
                         symbol = 'square', edge_color='green',  name='registered points') 
        
        
    def process_rois(self):
        pass


def check_if_suitable_layer(image, layer, roitype = 'registration'):
    
    assert image is not None, 'Image must be set.'
    assert layer is not None, 'ROIs must be set.'
    assert hasattr(image, 'data'), 'Image does not contain data.'
    assert hasattr(layer, 'data'), 'ROI does not contain data.'
    assert isinstance(layer, Points) or isinstance(layer, Labels), 'Specified ROI type is not accepted.'
    assert len(layer.data) > 0, 'At least one registration ROI must be chosen.'
    if roitype == 'registration':
        if isinstance(layer, Points):
            for point in layer.data:
                assert point[0] == 0.0, 'Registration ROIs must be in the first time frame of the image stack only.'
        if isinstance(layer, Labels):
            pass #TODO check that the labels are in the first frame
    if roitype == 'processing':
        points_in_first_frame = np.count_nonzero(layer.data[:,0]==0)
        time_frames_num = image.data.shape[0]
        assert len(layer.data) == time_frames_num*points_in_first_frame, 'The selected point are not registered. Please select registered points.'
        assert len(layer.data) % time_frames_num == 0, 'Check that roi_num is int, report if you get assertion error'
 

    
if __name__ == '__main__':
    
    nviewer = napari.Viewer()
    folder = os.getcwd() + "\\images"
    #folder = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\Plants2021\\WT_1'
    
    nviewer.open(folder)
    
    nviewer.layers[0].name = 'images'
    # add some labels
    img = nviewer.layers['images'].data
    
    s = img.shape
    test_label = np.zeros(s,dtype=int)
    test_label[0, 1037:1116, 801:880] = 1
    test_label[0, 761:800, 301:400] = 3
    # test_label[0, 900:1000, 700:900] = 4
    nviewer.add_labels(test_label, name='labels')
    # del(nviewer.layers['images'])
    # im0 = img[0,...]
    # im0_layer = nviewer.add_image(im0)
    
    # dask_list = []
    # for idx in range(20):
    #     new_list = [im for im in img]
    #     dask_list.append(im0)

    # stack = da.stack(dask_list, axis=0)
    # print(stack.shape)  # (nfiles, nz, ny, nx)
    # nviewer.layers['im0'].data = stack
    
    rv = RegistrationWidget(nviewer)
    nviewer.window.add_dock_widget(rv, name = 'test my widget')
    napari.run() 
    