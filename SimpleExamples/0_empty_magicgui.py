
import numpy as np
from napari.layers import Image
from magicgui import magicgui
import napari

@magicgui(call_button="apply threshold value")
def sum_images(image_layer :Image, 
               thresold: int = 120)->Image:
    
    data = np.array(image_layer.data)
    result = (data > thresold) *data
    return Image(result)


viewer = napari.Viewer()

viewer.window.add_dock_widget(sum_images,
                              name = 'Threshold widged')
napari.run()