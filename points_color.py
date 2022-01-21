# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 18:59:28 2022

@author: andrea
"""
import napari
import numpy as np
from skimage import data
viewer = napari.view_image(data.astronaut(), rgb=True)
points = np.array([[100, 100], [200, 200], [300, 100], [400,100]])

point_properties = {
    'color_idx': np.array([0, 1,2, 1]),
}

points_layer = viewer.add_points(
    points,
    properties=point_properties,
    edge_color='color_idx',
    edge_color_cycle= ['magenta', 'green', 'blue'],
    edge_width=5,
)

for idx in range(10):
    points_layer.current_properties = {'color_idx': f'{idx}'}
    points_layer.add(np.array([300,300+idx*20]))
#updates the properties

# new_point_properties = {
#     'color_idx': np.array([1, 0, 2, 1])
# }


# viewer.layers['points'].properties = new_point_properties