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
                 spinbox_step=0.05,
                 unit = '',
                 layout = None,
                 write_function = None,
                 read_function = None):
        
        self.name= name
        self._val = initial
        self.dtype = dtype
        self.spinbox_decimals = spinbox_decimals
        self.spinbox_step = spinbox_step
        self.unit = unit
        self.write_function = write_function
        # self.read_function = read_function
        self.create_spin_box(layout, dtype, vmin, vmax)
        
    def __repr__(self):
        return f'{self.name} : {self._val}'
    
        
    @property    
    def val(self):
        self._val = self.get_func()
        return self.dtype(self._val) 
    
    @val.setter 
    def val(self, new_val):
        new_val = self.dtype(new_val)
        self.set_func(new_val)
        self._val = new_val
        
    def create_spin_box(self, layout, dtype, vmin, vmax):
        name = self.name
        val = self._val
        if dtype == int:
            sbox = QSpinBox()
            sbox.setMaximum(vmax)
            sbox.setMinimum(vmin)
            self.set_func = sbox.setValue
            self.get_func = sbox.value
            change_func = sbox.valueChanged
        elif dtype == float:
            sbox = QDoubleSpinBox()
            sbox.setDecimals(self.spinbox_decimals)
            sbox.setSingleStep(self.spinbox_step)
            sbox.setMaximum(vmax)
            sbox.setMinimum(vmin)
            self.set_func = sbox.setValue
            self.get_func = sbox.value
            change_func = sbox.valueChanged
        elif dtype == bool:
            sbox = QCheckBox()
            self.set_func = sbox.setChecked
            self.get_func = sbox.checkState
            change_func = sbox.stateChanged
        
        else: raise(TypeError, 'Specified setting type not supported')
        
        self.set_func(val)
        if self.write_function is not None:
            change_func.connect(self.write_function)
        settingLayout = QFormLayout()
        settingLayout.addRow(sbox, QLabel(name))
        layout.addLayout(settingLayout)
        self.sbox = sbox


def add_timer(method):
    """Function decorator to mesaure the execution time of a method.
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self 
    """ 
    def inner(cls):
        print(f'\nStarting method "{method.__name__}" ...') 
        start_time = time.time() 
        result = method(cls) 
        end_time = time.time() 
        print(f'Execution time for method "{method.__name__}": {end_time-start_time:.6f} s') 
        return result
    inner.__name__ = method.__name__
    return inner 

def add_timer_to_function(function):
    """Function decorator to mesaure the execution time of a method.
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self 
    """ 
    def inner(*args):
        print(f'\nStarting method "{function.__name__}" ...') 
        start_time = time.time() 
        result = function(*args) 
        end_time = time.time() 
        print(f'Execution time for method "{function.__name__}": {end_time-start_time:.6f} s') 
        return result
    inner.__name__ = function.__name__
    return inner 


