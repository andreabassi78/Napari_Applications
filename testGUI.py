"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

from magicgui import magicgui
import napari
import numpy as np
from napari.layers import Image
from skimage import data

def create_widget(viewer):
    
    def extractor():
    
        @magicgui(call_button="Calculate", c={'bind':value})
        def f(c, x:int,y:int):
            print(c)
            return x+y
     
        return(5) 
     
        
    viewer.window.add_dock_widget(f, name = 'function', area='right')
    



@magicgui(call_button="Process ROIs")
def process_data_widget(image: Image):
    print(image.data)

value = 4



viewer = napari.view_image(data.astronaut(), rgb=True)

create_widget(viewer)

#viewer.window.add_dock_widget(process_data_widget, name = 'Platform')
