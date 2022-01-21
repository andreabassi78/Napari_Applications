# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:35:46 2022

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

from registration_utils import plot_data, save_in_excel, normalize_stack, select_rois_from_stack, select_rois_from_image
from registration_utils import align_with_registration, update_position
from registration_utils import calculate_spectrum, correct_decay 
import numpy as np
from magicgui import magicgui
import napari
from napari.layers import Image, Points, Labels
from skimage.measure import regionprops
import pathlib
import os
from napari.qt.threading import thread_worker
import warnings


def sum_projection(label_layer):
    '''
    Compresses a 3D label layer into a 2D array and returns the values
    '''
    values = np.asarray(label_layer.data).astype(int)
    if values.ndim>2:
        values = np.sum(values, axis=0)
    return(values)
    

def get_labels_values(labels_data):

    labels_props = regionprops(label_image=labels_data) # intensity_image=image0)    
    label_values = []
    for prop in labels_props:
        label_values.append(prop['label'])
    return label_values


def get_labels_color(labels_layer):
    colors_values = get_labels_values(labels_layer.data)
    colors = labels_layer.get_color(colors_values)
    return colors
    

@magicgui(call_button="Subtract background")
def subtract_background(image: Image, 
                        labels: Labels,
                        # result_name: str = 'corrected'
                        ):
    '''
    Subtracts a background from each plane of a stack image
    background is calulated as the mean intensity over one or more layers
    '''
    warnings.filterwarnings('ignore')
    result_name = image.name + '_corrected'
    
    def update_image(new_image):
        warnings.filterwarnings('ignore')
        try: 
            # if the layer exists, update the data
            viewer.layers[result_name].data = new_image
        except KeyError:
            # otherwise add it to the viewer
            viewer.add_image(new_image, name=result_name)
      
    @thread_worker(connect={'yielded': update_image})
    def _subtract_background():
        AMAX=2**16-1
        #check_if_suitable_layer(image, labels, roitype='registration')
        original = np.asarray(image.data)
        labels_data = sum_projection(labels)
        mask = np.asarray(labels_data>0).astype(int)
        corrected = np.zeros_like(original)
         
        for plane_idx, plane in enumerate(original.astype('int')):
            print(f'Correcting background on frame {plane_idx}')
            props = regionprops(label_image=mask,
                                intensity_image=plane)
            intensities = np.zeros(len(props))
            for roi_idx, prop in enumerate(props):
                intensities[roi_idx]=prop['mean_intensity']
            background = np.mean(intensities)
            diff = np.clip(plane-background, a_min=0, a_max=AMAX).astype('uint16')
            corrected[plane_idx,:,:] = diff
            yield corrected
    
    _subtract_background()

    
@magicgui(call_button="Register ROIs",
          roi_size = {'min':2})
def register_images(image: Image,
                    initial_rois: Labels,
                    roi_size: int = 100, 
                    median_filter_size:int = 3,
                    ):
    '''
    Registers rois chosen on image as square of side roi_size centered in the centroid of initial_rois 
    Based on cv2 registration.
    '''
    points_layer_name = f'registered rois {image.name}'
    # remove registration points if present
    label_colors = get_labels_color(initial_rois)
    for layer in viewer.layers:
        if layer.name == points_layer_name:
            viewer.layers.remove(points_layer_name)
    # initialize registration points layer with properties
    point_properties = {'color_idx': '0'}
    empty_point = Points([], ndim=image.data.ndim,
                 size = roi_size,
                 edge_width = roi_size//30+1, opacity = 1,
                 symbol = 'square',
                 properties = point_properties,
                 face_color = [1,1,1,0],
                 edge_color = 'color_idx',
                 edge_color_cycle = label_colors,
                 name = points_layer_name)
    viewer.layers.append(empty_point)
    
    
    def add_registered_points(centers):
        warnings.filterwarnings('ignore')
        for center_idx, center in enumerate(centers):
            registered_points = viewer.layers[points_layer_name]
            registered_points.add(center)
            registered_points.current_properties = {'color_idx': f'{center_idx}'}
            
      
    @thread_worker(connect={'yielded': add_registered_points})
    def _register_images():    
        
        print('Starting registration...')
        stack = np.asarray(image.data)
        normalize = True 
        if normalize: # this option is obsolete
            stack, _vmin, _vmax = normalize_stack(stack)
        image0 = stack[0,...]
        labels = sum_projection(initial_rois)
        initial_positions = []    
        props = regionprops(label_image=labels) # intensity_image=image0)
        for prop in props:
            t = 0
            yx=prop['centroid']
            initial_positions.append([t, yx[0], yx[1]])
        previous_rois = select_rois_from_image(image0, initial_positions, roi_size)
        time_frames_num, sy, sx = stack.shape
        next_positions = initial_positions.copy()
        for t_index in range(0, time_frames_num, 1):
            yield(next_positions)
            print(f'registering frame {t_index} of {time_frames_num-1}')
            next_rois = select_rois_from_image(stack[t_index,...], next_positions, roi_size)
            # registration based on opencv function
            aligned, original, dx, dy = align_with_registration(next_rois,
                                                                previous_rois,
                                                                median_filter_size,
                                                                roi_size
                                                                )
            next_positions = update_position(next_positions,
                                             dz = 1,
                                             dx_list = dx,
                                             dy_list = dy
                                             )
        print(f'... registered {time_frames_num} frames.') 
    _register_images()
    

def calculate_intensity(image:Image,
                        roi_num:int,
                        points_layer:Points,
                        labels_layer:Labels,
                        roi_size:int):
    """
    Calculates the mean intensity,
    within rectangular Rois of size roi_size, centered in points_layer,
    taking into account only the pixels that are in one of the labels of labels_layer
    """
    
    labels_data = sum_projection(labels_layer)
    label_values = get_labels_values(labels_data)
    stack = image.data
    locations = points_layer.data
    st, _sy, _sx = stack.shape
    rois = select_rois_from_stack(stack, locations, roi_size)
    label_rois = select_rois_from_image(labels_data, locations[0:roi_num], roi_size)
    intensities = np.zeros([st, roi_num])
    
    for time_idx in range(st):
        for roi_idx in range(roi_num):
            label_value = label_values[roi_idx]
            mask_indexes = label_rois[roi_idx] == label_value 
            global_idx = roi_idx + time_idx*roi_num
            roi = rois[global_idx]
            selected = roi[mask_indexes]
            intensity = np.mean(selected)
            intensities[time_idx, roi_idx] = intensity  
    
    return intensities


def measure_displacement(image, roi_num, points):
    """
    Measure the displacement of each roi:
    dr: relative to its position in the previous time frame 
    deltar: relative to the initial position.
    """
    stack = image.data
    st, sy, sx = stack.shape
    locations = points.data
    reshaped = locations.reshape((st, roi_num, stack.ndim))
    xy = reshaped[...,1:] # remove the time index value
    xy0 = xy[0,...] # take the x,y cohordinates of the rois in the first time frame
    
    deltar = np.sqrt( np.sum( (xy-xy0)**2, axis=2) )
    rolled = np.roll(xy, 1, axis=0) #roll one roi    
    rolled[0,...] = xy0
    dxy = xy-rolled
    dr = np.sqrt( np.sum( (dxy)**2, axis=2) )
    return xy, deltar, dxy, dr


@magicgui(call_button="Process registered ROIs")
def process_rois(image: Image, 
                 registered_points: Points,
                 labels: Labels,
                 correct_photobleaching: bool,
                 subroi_size:int = 100,
                 plot_results:bool = True,
                 save_results:bool = False,
                 path: pathlib.Path = os.getcwd()+"\\test.xlsx",
                 ):
    warnings.filterwarnings('ignore')
    print('Starting processing ...')
    time_frames_num, sy, sx = image.data.shape
    locations = registered_points.data
    roi_num = len(locations) // time_frames_num
    intensities = calculate_intensity(image, roi_num, registered_points, labels, subroi_size)
    yx, deltar, dyx, dr = measure_displacement(image, roi_num, registered_points)
    if correct_photobleaching:
        intensities = correct_decay(intensities)
    spectra = calculate_spectrum(intensities)    
        
    if plot_results:
        colors = get_labels_color(labels)
        plot_data(deltar, colors, "time index", "lenght (px)")
        plot_data(intensities, colors, "time index", "mean intensity")
        plot_data(spectra, colors, "frequency index", "power spectrum", plot_type = 'log')
    
    if save_results:
        save_in_excel(filename_xls = path,
                      sheet_name = 'Roi',
                      x = yx[...,1], 
                      y = yx[...,0],
                      length = deltar,
                      dx = dyx[...,1],
                      dy = dyx[...,0],
                      dr = dr,
                      intensity = intensities,
                      spectra = spectra
                      )
    print(f'... processed {time_frames_num} frames.')       
    
    
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    
    folder = os.getcwd() + "\\images"
    
    viewer.open(folder)
    viewer.layers[0].name = 'images'
   
    # add some labels
    image = viewer.layers['images'].data
    st,sy,sx = image.shape
    
    
    test_label = np.zeros([sy,sx],dtype=int)
    test_label[1069:1100, 831:850] = 1
    test_label[1370:1410, 626:640] = 7
   
    labels_layer = viewer.add_labels(test_label, name='labels')

    viewer.window.add_dock_widget(subtract_background, name = 'Subtract background',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(register_images, name = 'Registration',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(process_rois, name = 'Processing',
                                  area='right')
    warnings.filterwarnings('ignore')
    
    napari.run() 