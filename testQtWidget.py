# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:09:58 2022

@author: andrea
"""
import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
from skimage.measure import regionprops
from napari.layers import Image, Points, Labels,Shapes

import pandas as pd
from typing import Union
import numpy as np

class MyWidget(QWidget):
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        
        # initialize layout
        layout = QGridLayout()
        
        # add subtract background button
        subtract_btn = QPushButton('Subtract background', self)
        subtract_btn.clicked.connect(self.subtract_all)
        layout.addWidget(subtract_btn)
        
        # add normalize checkbox
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setChecked(False)
        layout.addWidget(self.normalize_checkbox)
        
        # add plot data checkbox
        self.plot_checkbox = QCheckBox("Plot")
        self.plot_checkbox.setChecked(True)
        layout.addWidget(self.plot_checkbox)
        
        # add normalize checkbox
        self.save_checkbox = QCheckBox("Save")
        self.save_checkbox.setChecked(False)
        layout.addWidget(self.save_checkbox)
        
        
        
        # activate layout
        self.setLayout(layout) # QWidget method
           
    def subtract_all(self):
        image = viewer.layers['images']
        labels = viewer.layers['labels']
        AMAX=2**16-1
        #check_if_suitable_layer(image, labels, roitype='registration')
        original = np.asarray(image.data).astype('int')
        corrected = np.zeros_like(original) 
        mask = np.asarray(labels.data[0,:,:]>0).astype(int)
        
        for plane_idx, plane in enumerate(original):
            props = regionprops(label_image=mask,
                                intensity_image=plane)
            intensities = np.zeros(len(props))
            for roi_idx, prop in enumerate(props):
                intensities[roi_idx]=prop['mean_intensity']
            background = np.mean(intensities)
            diff = np.clip(plane-background, a_min=0, a_max=AMAX)
            corrected[plane_idx,:,:] = diff.astype('uint16')
            
        self.viewer.add_image(corrected)    


    
if __name__ == '__main__':
   
    viewer = napari.Viewer()
    import os
    folder = os.getcwd() + "\\Registration\\images"
    
    viewer.open(folder)
    
    # add some labels
    image = viewer.layers['images'].data
    s = image.shape
    test_label = np.zeros(s,dtype=int)
    #test_label[0, 1037:1116, 801:880] = 1
    #test_label[0, 761:800, 301:400] = 3
    test_label[0, 900:1000, 700:900] = 4
    viewer.add_labels(test_label, name='labels')
    
    a = MyWidget(viewer)

    viewer.window.add_dock_widget(a, name = 'test my widget')
    napari.run() 