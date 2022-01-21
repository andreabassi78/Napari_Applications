# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:10:49 2022

@author: andrea
"""
import napari
import numpy as np
from napari.qt.threading import thread_worker
import time
from qtpy.QtWidgets import QPushButton
 
def update_layer(new_image):
         try:
             # if the layer exists, update the data
             viewer.layers['result'].data = new_image
         except KeyError:
             # otherwise add it to the viewer
             viewer.add_image(
                 new_image, contrast_limits=(0.45, 0.55), name='result'
             )

@thread_worker #(connect={'yielded': update_layer})
def large_random_images():
    cumsum = np.zeros((512, 512))
    time.sleep(0.5)
    for i in range(20):
        print(i)
        cumsum += np.random.rand(512, 512)
        yield cumsum / (i + 1)
        #time.sleep(0.1)

viewer = napari.Viewer()
worker = large_random_images()

worker.yielded.connect(update_layer)
worker.start()
napari.run()

