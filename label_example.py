# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 00:19:44 2022

@author: andrea
"""

import napari
from skimage.measure import regionprops, regionprops_table
import numpy as np
import os
import pandas
from skimage.measure import label
from napari.layers import Image, Points, Labels, Layer

viewer = napari.Viewer()

folder = os.getcwd()+"\\Registration\\images"

viewer.open(folder)

image = viewer.layers['images'].data

s = image.shape

test_label = np.zeros(s,dtype=int)
test_label[0, 0:150, 0:350] = 101
test_label[0, 351:450, 301:400] = 3



labels_layer = viewer.add_labels(test_label, name='labels')
# labels = labels_layer.data
labels = viewer.layers['labels'].data


# point0 = np.array([0,300,300])
# point1 = np.array([0,400,550])
# point3 = np.array([0,700,300])
# point4 = np.array([0,900,550])
# points = np.array([[[0,360,300],[0,390,300]],[[0,450,300],[0,560,300]]])
points = np.array([[0,360,300],[0,390,300],[0,450,300],[0,560,300]])

viewer.add_points(points)



props = ['label', 'mean_intensity', 'centroid']
props = []
table = regionprops_table(np.asarray(labels).astype(int),
                          intensity_image=np.asarray(image),
                          properties=props)
content = pandas.DataFrame(table)
print(content)

# props = regionprops(label_image=np.asarray(labels).astype(int),
#                     intensity_image=np.asarray(image))
# vals = []
# for prop in props:
#     vals.append(prop['mean_intensity'])

# print(vals)




