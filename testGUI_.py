"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

from magicgui import magicgui
import napari
import numpy as np
from napari.layers import Image
from skimage import data
from napari.layers.utils.stack_utils import images_to_stack



def process_one(nv: napari.Viewer, plane):
    return plane


@magicgui(call_button="process all")
def process_all(image: Image):
    stack = image.data
    
    viewer.
    
    for plane in stack:
        processed = process_one(viewer, plane)
        
 




#viewer = napari.view_image(data.astronaut(), rgb=True)
global viewer
viewer = napari.Viewer()
import os
folder = os.getcwd() + "\\Registration\\images"

viewer.open(folder)

viewer.window.add_dock_widget(process_all, name = 'Platform')
