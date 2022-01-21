"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

from magicgui import magicgui
import napari
import numpy as np
from napari.layers import Image
from skimage import data

@magicgui(call_button="process one")
def process_one(im:Image,index:int)->napari.types.LayerDataTuple:
    new_image_data = 2* im.data[index,...]
    print(index)
    return (new_image_data, {'name': 'MyImage'}, 'image')



@magicgui(call_button="process all")
def process_all(image: Image):
    
    
    
    for idx in range(3):
        process_one(image,idx)
 




#viewer = napari.view_image(data.astronaut(), rgb=True)

viewer = napari.Viewer()
import os
folder = os.getcwd() + "\\Registration\\images"

viewer.open(folder)

viewer.window.add_dock_widget(process_all, name = 'Platform')
viewer.window.add_dock_widget(process_one, name = 'Platform')
