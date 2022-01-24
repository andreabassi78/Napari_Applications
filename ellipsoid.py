# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:07:07 2022

@author: andrea
"""

import napari
import numpy as np
from skimage import data

# add the image
viewer = napari.view_image(data.camera(), name='photographer')
#viewer = napari.Viewer()

center = np.array([0,100,100])
deltar = 50
deltaz = 10


# create some ellipses
bbox1 = np.array([center+np.array([0,deltar,0]),
                  center+np.array([0,0,deltar]),
                  center-np.array([0,deltar,0]),
                  center-np.array([0,0,deltar])]
                 )

bbox2 = np.array([center+np.array([0,deltaz,0]),
                  center+np.array([deltaz,0,0]),
                  center-np.array([0,deltaz,0]),
                  center-np.array([deltaz,0,0])]
                 )

# put both shapes in a list
ellipses = [bbox1,bbox2]

# add an empty shapes layer
shapes_layer = viewer.add_shapes()

# add ellipses using their convenience method
shapes_layer.add_ellipses(
  ellipses, 
  edge_width=5,
  edge_color='coral', 
  face_color='royalblue'
)