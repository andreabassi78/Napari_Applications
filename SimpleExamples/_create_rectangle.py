# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:29:01 2022

@author: andrea
"""

import napari
import numpy as np
from skimage import data

def create_rectangle(center, sy, sx, color, name):
    cz=center[0]
    cy=center[1]
    cx=center[2]
    hsx = sx//2
    hsy = sy//2
    rectangle = [ [cz, cy+hsy, cx-hsx], # up-left
                  [cz, cy+hsy, cx+hsx], # up-right
                  [cz, cy-hsy, cx+hsx], # down-right
                  [cz, cy-hsy, cx-hsx]  # down-left
                  ]
    
    return rectangle
    

    
        
viewer = napari.view_image(data.astronaut(), rgb=True)
rectangle= create_rectangle([0,100,100], 50, 70, 'red', 'test') 
rectangle1= create_rectangle([0,200,100], 50, 70, 'red', 'test') 
rectangle2= create_rectangle([0,100,300], 50, 70, 'red', 'test') 

name = 'test'

viewer.add_shapes([np.array(rectangle)],
                          edge_width=2,
                          edge_color='red',
                          face_color=[1,1,1,0],
                          name = name
                          )
viewer.layers[name].add_rectangles(np.array([rectangle,rectangle1,rectangle2]), edge_color=np.array(['green','yellow','blue']))       