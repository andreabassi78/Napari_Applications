# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:02:27 2022

@author: andrea
"""

from napari.utils.events import Event
from napari import Viewer

def my_callback(event: Event):
    print("The number of dims shown is now:", event.value)

viewer = Viewer()
viewer.dims.events.ndim.connect(my_callback)