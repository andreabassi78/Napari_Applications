# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 00:16:58 2022

@author: andrea
"""
from psf_generator import PSF_simulator
from numpy import pi
from magicgui import magicgui
import napari
from napari.layers import Image, Points, Labels

um = 1.0
mm = 1000 * um
deg = pi/180


@magicgui(auto_call=True)
def set_values(NA:float = 0.7,
                wavelength:float = 0.5320*um,
                n:float = 1.00,
                Nxy:int = 127,
                Nz:int = 5,
                dxy:float = 0.05 * um,
                dz:float = 0.4 * um
                ):
    gen.NA = NA # Numerical aperture
    gen.n = n # refractive index at the object
    gen.wavelength = wavelength
    gen.Nxy = Nxy
    gen.Nz = Nz
    gen.dr = dxy
    gen.dz = dz
    
    print(gen.NA)

    
@magicgui(call_button="Add slab")    
def add_slab(n1:float = 1.51, # refractive index of the slab
             thickness:float = 170 * um, # slab thickness
             alpha:float = 0 * deg # angle of the slab relative to the y axis)    
             )->Image:
    set_values()  
    gen.generate_kspace()
    gen.add_slab_scalar(n1, thickness, alpha)
    gen.generate_pupil()
    gen.generate_3D_PSF()
    print(gen.write_name())
    return Image(gen.PSF3D,
                 name=gen.write_name(),
                 colormap='viridis')
    
    
@magicgui(call_button="Generate PSF")    
def generate_psf()->Image:
    set_values()  
    gen.generate_kspace()
    gen.generate_pupil()
    gen.generate_3D_PSF()
    # Show results    
    # gen.print_values()
    # gen.show_pupil()
    # gen.plot_phase()
    # gen.show_PSF_projections(aspect_ratio=1,
    #                      mode='sum') 
    gen.plot_psf_profile()
    
    print(gen.write_name())
    return Image(gen.PSF3D,
                 name=gen.write_name(),
                 colormap='viridis')
    
viewer = napari.Viewer()
gen = PSF_simulator() 

viewer.window.add_dock_widget((set_values, generate_psf), name = 'PSF generators',
                              area='right')
viewer.window.add_dock_widget(add_slab, name = 'Abberrations',
                              area='right')

napari.run()     