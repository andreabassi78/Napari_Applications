# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:47:14 2022

@author: andrea
"""

import napari
import numpy as np
from skimage import data

# create the list of polygons

def create_rectangle(center, sidey, sidex):
    cz=center[0]
    cy=center[1]
    cx=center[2]
    hsx = sidex//2
    hsy = sidey//2
    rectangle = [ [cz, cy+hsy, cx-hsx], # up-left
                  [cz, cy+hsy, cx+hsx], # up-right
                  [cz, cy-hsy, cx+hsx], # down-right
                  [cz, cy-hsy, cx-hsx]  # down-left
                ] 
    return np.array(rectangle)
    
# add the image
viewer = napari.view_image(data.camera(), name='photographer')
rect = create_rectangle([0,100,100],60,80)
# add the polygons
shapes_layer = viewer.add_shapes(rect, edge_width=2,
                          edge_color='green', face_color='white',)