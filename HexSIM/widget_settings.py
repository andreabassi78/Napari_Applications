# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:34:41 2022

@author: Andrea Bassi @ Polimi
"""
from qtpy.QtWidgets import QTableWidget, QSplitter, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QComboBox, QWidget, QFrame, QLabel, QFormLayout, QVBoxLayout, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox
import time

class Settings():
    ''' 
    Auxilliary class to create an object with a corresponding Qwidget,
    and update its value as a property (self.val)-
    - name of the QWidget (it contain a label)
    - dtype: Currently supported for int and float 
    - initial: initial value stored in the @property self.val
    - vmin, vmax: min and max values of the QWidget
    - layout: parent Qlayout    
    - read function: not implemented
    - write_function is executed on value change of the QWidget
    
    '''
    
    def __init__(self, name ='settings_name',
                 dtype = int,
                 initial = 0,
                 vmin = 0,
                 vmax = 2**16-1,
                 spinbox_decimals=3,
                 layout = None,
                 write_function = None,
                 read_function = None):
        
        self.name= name
        self._val = initial
        self.spinbox_decimals = spinbox_decimals
        self.write_function = write_function
        self.read_function = read_function
        self.create_spin_box(layout, dtype, vmin, vmax)
        
    @property    
    def val(self):
        self._val = self.sbox.value()
        return self._val 
    
    @val.setter 
    def val(self, new_val):
        self.sbox.setValue(new_val)
        self._val = new_val
        
    def create_spin_box(self, layout, dtype, vmin, vmax):
        name = self.name
        val = self._val
        if dtype == int:
            sbox = QSpinBox()
            sbox.setMaximum(vmax)
            sbox.setMinimum(vmin)
        elif dtype == float:
            sbox = QDoubleSpinBox()
            sbox.setDecimals(self.spinbox_decimals)
            sbox.setSingleStep(0.1)
            sbox.setMaximum(2**16-1)
        
        else: raise(TypeError, 'Specified setting type not supported')
        sbox.setValue(val)
        if self.write_function is not None:
            sbox.valueChanged.connect(self.write_function)
        settingLayout = QFormLayout()
        settingLayout.addRow(QLabel(name), sbox)
        layout.addLayout(settingLayout)
        self.sbox = sbox


def add_timer(function):
    """Function decorator to mesaure the execution time of a method.
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self 
    """ 
    def inner(cls):
        print(f'\nStarting method "{function.__name__}" ...') 
        start_time = time.time() 
        result = function(cls) 
        end_time = time.time() 
        print(f'Execution time for method "{function.__name__}": {end_time-start_time:.6f} s') 
        return result
    inner.__name__ = function.__name__
    return inner 

    
def add_update_display(function):
    """Function decorator to to update display at the end of the execution
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self 
    """ 
    def inner(cls):
        result = function(cls)
        cls.update_display()
        return result
    inner.__name__ = function.__name__
    return inner  
