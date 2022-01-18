# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:35:46 2022

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

from registration_utils import plot_data, save_in_excel, normalize_stack, select_rois
from registration_utils import align_with_registration, update_position
from registration_utils import calculate_spectrum, correct_decay 

import numpy as np
from magicgui import magicgui
import napari
from napari.layers import Image, Points, Labels,Shapes
from skimage.measure import regionprops
import pathlib
import os


@magicgui(call_button="Subtract background")
def subtract_background(image: Image, 
                        labels: Labels,
                        )->Image:
    AMAX=2**16-1
    check_if_suitable_layer(image, labels, roitype='registration')
    original = np.asarray(image.data).astype('float32')
    corrected = np.zeros_like(original) 
    label_plane = labels.data[0,:,:]
    for plane_idx, plane in enumerate(original):
        props = regionprops(label_image=np.asarray(label_plane).astype(int),
                            intensity_image=plane)
        intensities = np.zeros(len(props))
        for idx,prop in enumerate(props):
            intensities=prop['mean_intensity']
        background = np.mean(intensities)
        
        diff = np.clip(plane-background, a_min=0, a_max=AMAX)
        corrected[plane_idx,:,:] = np.uint16(diff)    
    return Image(corrected)

    
@magicgui(call_button="Register ROIs",
          roi_size = {'min':2})
def register_images(image: Image, 
                    initial_rois: Labels,
                    roi_size: int = 100,  
                    median_filter_size:int = 3, 
                    time_series_undersampling:int =1 # TODO remove after testing
                    )-> Points:
    
    check_if_suitable_layer(image,initial_rois, roitype='registration')

    stack = image.data

    props = regionprops(label_image=np.asarray(initial_rois.data).astype(int),
                            intensity_image=np.asarray(stack))
    centroids = []
    for prop in props:
        centroids.append(prop['centroid'])
    initial_points =  Points(centroids)
    
    normalize = True 
    if normalize: # this option is obsolete
        stack, _vmin, _vmax = normalize_stack(stack)
    
    time_frames_num, sy, sx = stack.shape
    initial_position_list = np.flip(initial_points.data).tolist() #TODO change position list using napari points notation
    position_list = initial_position_list
    initial_img = stack[0, ...]        
    
    registered_points = []
    for pos in initial_position_list:
        pointdata=[0,pos[1],pos[0]]
        registered_points.append(pointdata)
 
    for t_index in range(1, time_frames_num, time_series_undersampling):
        
        #previous_img = select_image(t_index, vmin, vmax)
        next_img = stack[t_index, ...]
        
        previous_rois = select_rois(initial_img, initial_position_list, roi_size)
        next_rois = select_rois(next_img, position_list, roi_size)
        
        # registration
        aligned, original, dx, dy = align_with_registration(next_rois,
                                                            previous_rois,
                                                            median_filter_size,
                                                            roi_size)
    
        position_list, _length = update_position(position_list, initial_position_list, dx, dy)
        
        for pos in position_list:
            pointdata=[t_index,pos[1],pos[0]]
            registered_points.append(pointdata)

    centers = Points(registered_points, size=roi_size, edge_width=roi_size//25+1, opacity = 0.3,
                     symbol = 'square', edge_color='green',  name='registered points')   

    return centers  


@magicgui(call_button="Draw rectangular ROIs")
def _draw_squares(points_layer:Points, roi_size=100)->Shapes:
    from registration_utils import rectangle
    rectangles = []
    for center in points_layer.data:
        rectangles.append(rectangle(center, roi_size, roi_size)) 
    return Shapes(rectangles)    
    

def calculate_intensity(image, roi_num, points_layer, labels_layer, roi_size):
    """
    Calculates the mean intensity,
    of Roi designed around each point in points
    """
    stack = image.data
    locations = points_layer.data
    label_stack = labels_layer.data
    st, _sy, _sx = stack.shape
    intensities = np.zeros([st,roi_num])
    reshaped_locations = locations.reshape((st, roi_num, stack.ndim))
    xy_locations = reshaped_locations[...,1:] # remove the time index value 
    label_imgs = label_stack[0,...]
    print(label_imgs.shape)
    
    for time_idx in range(st):
        imgs = stack[time_idx,...]
        position_list = np.flip(xy_locations[time_idx,...,]).tolist()
        rois = select_rois(imgs, position_list, roi_size) 
        label_rois = select_rois(label_imgs, position_list, roi_size)
        for roi_idx, roi in enumerate(rois):
            label_mask = np.uint16(label_rois[roi_idx]>0)
            intensity = np.mean(roi*label_mask)
            intensities[time_idx, roi_idx] = intensity
    return intensities


def measure_displacement(image, roi_num, points):
    """
    Measure the displacement of each roi:
    dr: relative to its position in the previous time frame 
    deltar: relative to the initial position.
    """
    stack = image.data
    locations = points.data
    st, sy, sx = stack.shape
    
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
    xy, deltar, dxy, dr = measure_displacement(image, roi_num, registered_points)
    if correct_photobleaching:
        intensities = correct_decay(intensities)
    spectra = calculate_spectrum(intensities)    
        
    if plot_results:   
        plot_data(deltar, "time index", "lenght (px)")
        plot_data(intensities, "time index", "mean intensity")
        plot_data(spectra, "frequency index", "power spectrum", plot_type = 'log')
       
    if save_results:
        save_in_excel(filename_xls = path, 
                      sheets_number = roi_num,
                      y = xy[:,0], #TODO check this values and indexing
                      x = xy[:,1],
                      length = deltar,
                      dy = dxy[:,0],
                      dx = dxy[:,1],
                      dr = dr,
                      intensity = intensities,
                      )    


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
    test_label[0, 1037:1116, 801:880] = 135
    test_label[0, 761:800, 301:400] = 35
    labels_layer = viewer.add_labels(test_label, name='labels')

    viewer.window.add_dock_widget(subtract_background, name = 'Subtract background', area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(register_images, name = 'Registration', area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(process_rois, name = 'Processing', area='right')
    
    napari.run() 
    
    