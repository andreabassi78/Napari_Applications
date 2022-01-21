# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:18:22 2022

@author: andrea
"""

import numpy as np
from magicgui import magicgui
import napari
from napari.layers import Image, Points, Labels
from skimage.measure import regionprops
import pathlib
import os
from napari.qt.threading import thread_worker
import warnings
viewer = napari.Viewer()

folder = os.getcwd() + "\\Registration\\images"
#folder = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Data\\Plants2021\\WT_1'

viewer.open(folder)
viewer.layers[0].name = 'images'

#add some points
# points = np.array([[0,1076, 829], [0,1378, 636]])
# points_layer = viewer.add_points(
#     points,
#     size=20,
#     name= 'selected points')

# add some labels
image = viewer.layers['images'].data
st,sy,sx = image.shape

# test_label = np.zeros([st,sy,sx],dtype=int)
# test_label[0,1037:1116, 801:880] = 1
# test_label[0,761:800, 301:400] = 3
# test_label[0,761:800, 501:600] = 4

test_label = np.zeros([sy,sx],dtype=int)
test_label[1037:1116, 801:880] = 1
test_label[1306:1450, 606:660] = 3
 
labels_layer = viewer.add_labels(test_label, name='labels')

label_colors = np.array( [ [0,1,0],[0,1,1]])


empty_point = Points([], ndim =3, size=100,
                edge_width=100//25+1, opacity = 0.3,
                symbol = 'square', 
                edge_color_cycle = label_colors,
                name='test') 

viewer.layers.append(empty_point)

centers = np.array(  [[0,100,100],[0,100,100]])

viewer.layers['test'].add(centers)

viewer.layers['test'].set

point_properties = {
    'good_point': np.array([True, True, False]),
    'confidence': np.array([0.99, 0.8, 0.2]),
}


# for layer in viewer.layers:
#     if layer.name == 'nome':
#         layer.data = []
#         del layer
# viewer.add_points([0,0,0], size=100,
#              properties = {'edge_color_mode': 'CYCLE'},     
#              edge_width=100//25+1, opacity = 0.3,
#              symbol = 'square', 
#              edge_color_cycle = label_colors,
#              name=pointsname)
    
# def add_registered_points(centers):
#     warnings.filterwarnings('ignore')
    
#     viewer.layers[pointsname].add(centers)