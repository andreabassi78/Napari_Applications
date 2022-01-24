# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 00:16:58 2022

@author: Andrea Bassi @Polimi
"""
from psf_generator import PSF_simulator
import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QComboBox, QWidget, QFrame, QLabel, QFormLayout, QVBoxLayout, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox
from skimage.measure import regionprops
from napari.layers import Image, Points, Labels, Shapes
import numpy as np


class Settings():
    def __init__(self, name ='settings_name',
                 dtype = int,
                 initial_value = 0,
                 vmin = 0,
                 vmax = 2**16-1,
                 layout = None,
                 write_function = None,
                 read_function = None):
        
        self.name= name
        self._val = initial_value
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
            sbox.setDecimals(3)
            sbox.setSingleStep(0.1)
            sbox.setMaximum(2**16-1)
        
        else: raise(TypeError, 'Specified setting type not supported')
        sbox.setValue(val)
        if self.write_function is not None:
            sbox.valueChanged.connect(self.write_function)
        settingLayout = QFormLayout()
        settingLayout.addRow(name, sbox)
        layout.addLayout(settingLayout)
        self.sbox = sbox
        

class Psf_widget(QWidget):
    
    ABERRATION_DICT = {0:'No aberrations', 1:'Slab', 2:'Zernike'}
    
    def __init__(self, viewer:napari.Viewer,
                 NA:float = 0.5,
                 n:float = 1.00,
                 wavelength:float = 0.5320,
                 Nxy:int = 127,
                 Nz:int = 63,
                 dxy:float = 0.10,
                 dz:float = 0.2
                                 ):
        self.viewer = viewer
        super().__init__()
        self.setup_ui() # run setup_ui before instanciating the Settings
        self.create_Settings(NA, n, wavelength, Nxy, Nz, dxy, dz)
        self.create_aberration_Settings()
        self.initialize_simulator()
        
    
    def initialize_simulator(self):
        self.gen = PSF_simulator(self.NA.val, self.n.val, self.wavelength.val,
                      self.Nxy.val , self.Nz.val, dr = self.dxy.val, dz = self.dz.val)
        self.gen.generate_kspace()
        
        active_aberration = self.aberration_combo.currentIndex()
        self.add_aberration(active_aberration)
             
    
    def setup_ui(self):     
        # initialize layout
        layout = QVBoxLayout()
        # prepare sub_layout for Settings
        settings_layout = QVBoxLayout()
        layout.addLayout(settings_layout)
        
        # add plot data checkbox
        self.plot_checkbox = QCheckBox("Plot PSF profile")
        self.plot_checkbox.setChecked(False)
        layout.addWidget(self.plot_checkbox)
        # add show stack and show MIP checkbox
        self.stack_checkbox = QCheckBox("Show stack")
        self.stack_checkbox.setChecked(False)
        layout.addWidget(self.stack_checkbox)
        # add show stack and show MIP checkbox
        self.projections_checkbox = QCheckBox("Show projections")
        self.projections_checkbox.setChecked(True)
        layout.addWidget(self.projections_checkbox)
        self.mip_checkbox = QCheckBox("Show MIP")
        self.mip_checkbox.setChecked(False)
        layout.addWidget(self.mip_checkbox)
        
        # create reference to layout for the Settings
        self.settings_layout = settings_layout
        # Aberrations
        self.aberration_combo = QComboBox()
        self.aberration_combo.addItems(list(self.ABERRATION_DICT.values()))
        self.aberration_combo.currentIndexChanged.connect(self.initialize_simulator)
        
        layout.addWidget(self.aberration_combo)
        aberrations_frame = QFrame()
        aberrations_frame.setFrameShape(QFrame.StyledPanel)
        #aberrations_frame.setFrameShadow(QFrame.Plain)
        #aberrations_frame.setLineWidth(1)
        #aberrations_frame.setStyleSheet('border-color: rgb(50,50,60); border-style: outset; border-width: 2px,')
        aberrations_frame.setStyleSheet('background-color: rgb(50,50,60)')
        aberration_layout = QVBoxLayout(aberrations_frame)
        layout.addWidget(aberrations_frame)
        
        # create reference to aberration layout for the  Settings
        self.aberration_layout = aberration_layout
        # add calculate psf button
        calculate_btn = QPushButton('Calculate PSF')
        calculate_btn.clicked.connect(self.calculate_psf)
        layout.addWidget(calculate_btn)
        # reset_btn = QPushButton('Reset')
        # reset_btn.clicked.connect(self.initialize_simulator)
        # layout.addWidget(reset_btn)
        # activate layout
        self.setLayout(layout) # QWidget method
    
    
    def create_Settings(self, NA, n, wavelength, Nxy, Nz, dxy, dz):
        self.NA = Settings(name = 'NA', dtype = float, initial_value= NA,
                           layout = self.settings_layout,
                           write_function = self.initialize_simulator)
        self.n = Settings(name = 'n', dtype = float, initial_value=n,
                            layout = self.settings_layout,
                            write_function = self.initialize_simulator)
        self.wavelength = Settings(name='wavelength', dtype = float, initial_value=wavelength,
                            layout = self.settings_layout,
                            write_function = self.initialize_simulator)
        self.Nxy = Settings(name ='N', dtype = int, initial_value = Nxy, vmin = 1,
                            layout = self.settings_layout,
                            write_function = self.initialize_simulator)
        self.Nxy.sbox.setSingleStep(2) # Nxy must be odd
        self.Nz = Settings(name ='Nz', dtype = int, initial_value = Nz, vmin = 1,
                            layout = self.settings_layout,
                            write_function = self.initialize_simulator)
        self.Nxy.sbox.setSingleStep(2) # Nz must be odd
        self.dxy = Settings(name ='dx/dy', dtype = float, initial_value = dxy,
                            layout = self.settings_layout,
                            write_function = self.initialize_simulator)
        self.dz = Settings(name = 'dz', dtype = float, initial_value = dz,
                            layout = self.settings_layout,
                            write_function = self.initialize_simulator)
        
        
    def create_aberration_Settings(self):
        self.n1 = Settings(name = 'n1', dtype = float, initial_value = 1.51, 
                            layout = self.aberration_layout,
                            write_function = self.initialize_simulator)
        self.thickness = Settings(name = 'slab thickness (um)', dtype = float, initial_value= 100.0, 
                            layout = self.aberration_layout,
                            write_function = self.initialize_simulator)
        self.alpha = Settings(name = 'angle (deg)', dtype = float, initial_value= 0.0, 
                            layout = self.aberration_layout,
                            write_function = self.initialize_simulator)
                 
        self.N = Settings(name = 'N', dtype = int, initial_value = 3,
                            layout = self.aberration_layout,
                            write_function = self.initialize_simulator)
        self.M = Settings(name = 'M', dtype = int, initial_value= 1,
                            layout = self.aberration_layout,
                            write_function = self.initialize_simulator)
        self.weight = Settings(name = 'weight (lambdas)', dtype = float, initial_value= 1.0,
                            layout = self.aberration_layout,
                            write_function = self.initialize_simulator)
                
    
    def add_aberration(self, value):
        if value == 1:
            self.gen.add_slab_scalar(self.n1.val, self.thickness.val, self.alpha.val)
        if value == 2:
            self.gen.add_Zernike_aberration(self.N.val, self.M.val, self.weight.val)
        
    
    def calculate_psf(self):
        self.gen.generate_pupil()
        self.gen.generate_3D_PSF()
        
        if self.plot_checkbox.checkState():
            self.gen.plot_psf_profile()
            
        if self.projections_checkbox.checkState():
            self.show_PSF_projections()
            
        if self.stack_checkbox.checkState():
            self.viewer.add_image(self.gen.PSF3D,
                         name=self.gen.write_name(),
                         colormap='twilight')
        
        
    def show_PSF_projections(self): 
        PSF = self.gen.PSF3D
        if self.mip_checkbox.checkState():
            # create maximum intensity projection
            im_xy = np.amax(PSF, axis=0)
            im_xz = np.amax(PSF, axis=1)
            text = 'mip'
        else:
            Nz,Ny,Nx = PSF.shape
            im_xy = PSF[Nz//2:,:]
            im_xz = PSF[:,Ny//2,:]
            text = 'plane'
            
        imageXZ = self.viewer.add_image(im_xz,
                     name=f'xz_{text}_{self.gen.write_name()}',
                     colormap='twilight')
        imageXZ.scale = (self.dz.val/self.dxy.val, 1)
        
        self.viewer.add_image(im_xy,
                     name=f'xy_{text}_{self.gen.write_name()}',
                     colormap='twilight')
        
        
        
if __name__ == '__main__':
   
    viewer = napari.Viewer()
    widget = Psf_widget(viewer)
    viewer.window.add_dock_widget(widget,
                                  name = 'PSF Simulator @Polimi',
                                  add_vertical_stretch = True)
    napari.run()      