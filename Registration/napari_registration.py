# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:35:46 2022

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

from registration_utils import plot_data, save_in_excel, normalize_stack, select_rois_from_stack
from registration_utils import align_with_registration, update_position
from registration_utils import calculate_spectrum, correct_decay 
import numpy as np
from magicgui import magicgui
import napari
from napari.layers import Image, Points, Labels,Shapes
from skimage.measure import regionprops
import pathlib
import os
import time


def time_it(func):
    def inner(*args):
        t0 = time.time()
        result = func(*args)
        print(f'Time for "{func.__name__}": {time.time()-t0:.2f} s')
        return result
    return inner
    

def toc(t0, action = 'initializing'):
    t1 = time.time()
    print(f'Time for {action}: {t1-t0}')
    return t1


@magicgui(call_button="Subtract background")
def subtract_background(image: Image, 
                        labels: Labels,
                        )->Image:
    AMAX=2**16-1
    check_if_suitable_layer(image, labels, roitype='registration')
    original = np.asarray(image.data).astype('int')
    corrected = np.zeros_like(original) 
    mask = np.asarray(labels.data[0,:,:]>0).astype(int)
    
    for plane_idx, plane in enumerate(original):
        props = regionprops(label_image=mask,
                            intensity_image=plane)
        intensities = np.zeros(len(props))
        for roi_idx, prop in enumerate(props):
            intensities[roi_idx]=prop['mean_intensity']
        background = np.mean(intensities)
        diff = np.clip(plane-background, a_min=0, a_max=AMAX)
        corrected[plane_idx,:,:] = diff.astype('uint16')
        
    return Image(corrected)

   
@magicgui(call_button="Register ROIs",
          roi_size = {'min':2})
def register_images(image: Image, 
                    initial_rois: Labels,
                    roi_size: int = 100,  
                    median_filter_size:int = 3, 
                    time_series_undersampling:int =1 # TODO remove after testing
                    )-> Points:
    t0 = time.time()
    check_if_suitable_layer(image,initial_rois, roitype='registration')

    stack = image.data

    props = regionprops(label_image=np.asarray(initial_rois.data).astype(int),
                            intensity_image=np.asarray(stack))
    initial_positions = []
    for prop in props:
        initial_positions.append(prop['centroid'])
    
    normalize = True 
    if normalize: # this option is obsolete
        stack, _vmin, _vmax = normalize_stack(stack)
    
    previous_rois = select_rois_from_stack(stack, initial_positions, roi_size)
    registered_positions = []
    next_positions = initial_positions.copy()
    registered_positions.append(next_positions)
    
    roi_num = len(initial_positions)
    time_frames_num, sy, sx = stack.shape
    
    print('Time for roi extraction:', time.time()-t0)
    t0 = time.time()
    
    for t_index in range(1, time_frames_num, time_series_undersampling):

        next_rois = select_rois_from_stack(stack, next_positions, roi_size)
        # registration based on opencv function
        aligned, original, dx, dy = align_with_registration(next_rois,
                                                            previous_rois,
                                                            median_filter_size,
                                                            roi_size
                                                            )
        next_positions = update_position(next_positions,
                                         dz = time_series_undersampling,
                                         dx_list = dx,
                                         dy_list = dy
                                         )
        registered_positions.append(next_positions)
    
    # positions must be reshaped in a single list, to be converted in napari points 
    reshaped_positions = np.reshape(np.array(registered_positions),
                                    [time_frames_num*roi_num, stack.ndim])
    centers = Points(reshaped_positions, size=roi_size, edge_width=roi_size//25+1, opacity = 0.3,
                     symbol = 'square', edge_color='green',  name='registered points')
    
    print('Time for registration:', time.time()-t0)
    t0 = time.time()
    
    print(f'Registered {time_frames_num//time_series_undersampling} frames. ')
    return centers  


@magicgui(call_button="Draw rectangular ROIs")
def _draw_squares(points_layer:Points, roi_size=100)->Shapes:
    from registration_utils import rectangle
    rectangles = []
    for center in points_layer.data:
        rectangles.append(rectangle(center, roi_size, roi_size)) 
    
    return Shapes(rectangles)    
    
@time_it
def calculate_intensity(image, roi_num, points_layer, labels_layer, roi_size):
    """
    Calculates the mean intensity,
    of Roi designed around each point in points
    """
    stack = image.data
    locations = points_layer.data
    label_stack = labels_layer.data
    st, _sy, _sx = stack.shape
    
    intensities = np.zeros([st, roi_num])
    rois = select_rois_from_stack(stack, locations, roi_size)
    label_rois = select_rois_from_stack(label_stack, locations[0:roi_num], roi_size)
    
    for time_idx in range(st):
        for roi_idx in range(roi_num):
            label_mask = np.uint16(label_rois[roi_idx]>0)
            global_idx = time_idx + time_idx*roi_idx
            roi = rois[global_idx]
            intensity = np.mean(roi*label_mask)
            intensities[time_idx, roi_idx] = intensity
    
    return intensities

@time_it
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
    
    check_if_suitable_layer(image, registered_points, roitype='processing')
    
    time_frames_num, sy, sx = image.data.shape
    locations = registered_points.data
    roi_num = len(locations) // time_frames_num
    
    intensities = calculate_intensity(image, roi_num, registered_points, labels, subroi_size)
    yx, deltar, dyx, dr = measure_displacement(image, roi_num, registered_points)
    if correct_photobleaching:
        intensities = correct_decay(intensities)
    spectra = calculate_spectrum(intensities)    
        
    if plot_results:   
        plot_data(deltar, "time index", "lenght (px)")
        plot_data(intensities, "time index", "mean intensity")
        # plot_data(spectra, "frequency index", "power spectrum", plot_type = 'log')
    
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
    print(f'Processed {time_frames_num} frames.')       


def check_if_suitable_layer(image, layer, roitype = 'registration'):
    
    assert image is not None, 'Image must be set.'
    assert layer is not None, 'ROIs must be set.'
    assert hasattr(image, 'data'), 'Image does not contain data.'
    assert hasattr(layer, 'data'), 'ROI does not contain data.'
    assert isinstance(layer, Points) or isinstance(layer, Labels), 'Specified ROI type is not accepted.'
    assert len(layer.data) > 0, 'At least one registration ROI must be chosen.'
    if roitype == 'registration':
        if isinstance(layer, Points):
            for point in layer.data:
                assert point[0] == 0.0, 'Registration ROIs must be in the first time frame of the image stack only.'
        if isinstance(layer, Labels):
            pass #TODO check that the labels are in the first frame
    if roitype == 'processing':
        points_in_first_frame = np.count_nonzero(layer.data[:,0]==0)
        time_frames_num = image.data.shape[0]
        assert len(layer.data) == time_frames_num*points_in_first_frame, 'The selected point are not registered. Please select registered points.'
        assert len(layer.data) % time_frames_num == 0, 'Check that roi_num is int, report if you get assertion error'
 
    
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    
    folder = os.getcwd() + "\\images"
    
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

    viewer.window.add_dock_widget(subtract_background, name = 'Subtract background', area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(register_images, name = 'Registration', area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(process_rois, name = 'Processing', area='right')
    napari.run() 
    # register_images()
    # process_rois(plot_results=True,save_results=False)