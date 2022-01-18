# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:09:58 2022

@author: andrea
"""
import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog

import pandas as pd
from typing import Union
import numpy as np


class MyWidget(QWidget):
    """
    The table widget represents a table inside napari.
    Tables are just views on `properties` of `layers`.
    """
    def __init__(self, viewer: napari.viewer):
        super().__init__()
        self._viewer = viewer
        
        self._view = QTableWidget()
        
        correct_button = QPushButton("Correct background")
        correct_button.clicked.connect(self._correct)
        self.setWindowTitle("Correct background")
        self.setLayout(QGridLayout())
        action_widget = QWidget()
        action_widget.setLayout(QHBoxLayout())
        action_widget.layout().addWidget(correct_button)
        
    def _correct(self):
        pass

    
if __name__ == '__main__':
    
    
    
    viewer = napari.Viewer()
    import os
    folder = os.getcwd() + "\\Registration\\images"
    
    viewer.open(folder)
    
    # add some points
    # points = np.array([[0,1076, 829], [0,1378, 636]])
    # points_layer = viewer.add_points(
    #     points,
    #     size=20,
    #     name= 'selected points')
    
    # add some labels
    image = viewer.layers['images'].data
    s = image.shape
    test_label = np.zeros(s,dtype=int)
    test_label[0, 1037:1116, 801:880] = 1
    test_label[0, 761:800, 301:400] = 3
    test_label[0, 761:800, 501:600] = 4
    labels_layer = viewer.add_labels(test_label, name='labels')
    
    a = MyWidget(labels_layer)

    viewer.window.add_dock_widget(a, name = 'test my widget')
    napari.run() 