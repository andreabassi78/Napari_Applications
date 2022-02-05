# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:35:46 2022

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

from registration_utils import plot_data, save_in_excel, normalize_stack, select_rois_from_stack, select_rois_from_image
from registration_utils import align_with_registration, update_position, resize_stack,rescale_position
from registration_utils import calculate_spectrum, correct_decay, stack_registration,apply_warp_to_stack
import numpy as np
from magicgui import magicgui
import napari
from napari.layers import Image, Points, Labels
from skimage.measure import regionprops
import pathlib
import os
from napari.qt.threading import thread_worker
import warnings


def max_projection(label_layer):
    '''
    Compresses a 3D label layer into a 2D array and returns the values.
    Selects the label with the highest value in case of overlap.
    '''
    values = np.asarray(label_layer.data).astype(int)
    if values.ndim>2:
        
        values = np.max(values, axis = 0)
        
    return(values)
    

def get_labels_values(labels_data):

    labels_props = regionprops(label_image=labels_data) # intensity_image=image0)    
    label_values = []
    for prop in labels_props:
        label_values.append(prop['label'])
    return label_values


def get_labels_color(labels_data):
    colors_values = get_labels_values(labels_data)
    labels_layer = Labels(labels_data) # TODO use an active label on viewer
    colors = labels_layer.get_color(colors_values)
    return colors


@magicgui(call_button = "Set optimal contrast")
def optimize_contrast(image: Image, saturation : float = 1):
    original = image.data
    vmax = np.amax(original)
    vmin = np.amin(original)
    image.contrast_limits = (int(vmin), int(vmax/saturation))
    

@magicgui(call_button="Subtract background")
def subtract_background(image: Image, 
                        labels: Labels,
                        # result_name: str = 'corrected'
                        ):
    '''
    Subtracts a background from each plane of a stack image
    background is calulated as the mean intensity over one or more layers
    '''
    
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
        warnings.filterwarnings('ignore')
        AMAX=2**16-1
        #check_if_suitable_layer(image, labels, roitype='registration')
        original = np.asarray(image.data)
        labels_data = max_projection(labels)
        mask = np.asarray(labels_data>0).astype(int)
        corrected = np.zeros_like(original)
         
        for plane_idx, plane in enumerate(original.astype('int')):
            print(f'Correcting background on frame {plane_idx}')
            props = regionprops(label_image=mask,
                                intensity_image=plane)
            background = props[0]['mean_intensity'] #only one region is defined in mask
            diff = np.clip(plane-background, a_min=0, a_max=AMAX).astype('uint16')
            corrected[plane_idx,:,:] = diff
            yield corrected
        subtract_background.enabled = True
         
    
    subtract_background.enabled = False
    _subtract_background()


@magicgui(call_button="Register stack",
          mode={"choices": ['Translation','Affine','Euclidean','Homography']})
def register_stack(image: Image,
                   labels_layer: Labels,
                   mode: str = 'Translation',
                   register_entire_image:bool = False
                   ):
    register_stack.enabled = False
    labels = max_projection(labels_layer)
    t_idx = viewer.dims.current_step[0]
    position, roi_sy, roi_sx = get_rois_props(labels, t_idx) 
    rectangle = create_rectangles(position, roi_sy, roi_sx)
    viewer.add_shapes(np.array(rectangle),
                      edge_width=2,
                      edge_color='red',
                      face_color=[1,1,1,0],
                      name = f'roi_{image.name}'
                      )
    
    def show_registered_stack(params):
        (stk,ridx) = params
        registered_name= f'registered_{image.name}_roi{ridx}'
        viewer.add_image(stk, name= registered_name)
        register_stack.enabled = True
    
    @thread_worker(connect={'yielded':show_registered_stack})
    def _register_stack():
        warnings.filterwarnings('ignore') 
        for roi_idx, pos in enumerate(position):
            y = int(pos[-2])
            x = int(pos[-1])
            half_sizey = roi_sy[roi_idx]//2
            half_sizex = roi_sx[roi_idx]//2
            stack = np.asarray(image.data)
            st,sy,sx = stack.shape
            substack = stack[:, y-half_sizey:y+half_sizey,
                              x-half_sizex:x+half_sizex]
            registered, wm = stack_registration(substack, t_idx, mode = mode)
    
            if register_entire_image: # registers the largest possible frame around y,x
                max_half_sizey = min(sy-y,y)
                max_half_sizex = min(sx-x,x)
                max_substack = stack[:, y-max_half_sizey:y+max_half_sizey,
                                  x-max_half_sizex:x+max_half_sizex]
                registered = apply_warp_to_stack(max_substack, wm)
    
            yield (registered,roi_idx) 
    
    _register_stack()
    
    
def get_rois_props(label_data, t=0, bbox_zoom = 1):
    centroids = []  
    roi_sizes_x = []
    roi_sizes_y = []
    props = regionprops(label_image=label_data)#, intensity_image=image0)
    for prop in props:
        yx = prop['centroid']
        bbox = prop['bbox']
        _sizey = int(np.abs(bbox[0]-bbox[2])*bbox_zoom)
        _sizex = int(np.abs(bbox[1]-bbox[3])*bbox_zoom)  
        centroids.append([t, yx[0], yx[1]])
        roi_sizes_y.append(_sizey)
        roi_sizes_x.append(_sizex)
    return centroids, roi_sizes_y, roi_sizes_x   


def create_point(center, name):
    try:
        viewer.layers[name].add(np.array(center))
    except:
        viewer.add_points(np.array(center),
                          edge_color='green',
                          face_color=[1,1,1,0],
                          name = name
                          )


def create_rectangles(centers, sys, sxs):
    rectangles = []
    for center,sy,sx in zip(centers,sys,sxs):
        cz=center[0]
        cy=center[1]
        cx=center[2]
        hsx = sx//2
        hsy = sy//2
        rectangle = [ [cz, cy+hsy, cx-hsx], # up-left
                      [cz, cy+hsy, cx+hsx], # up-right
                      [cz, cy-hsy, cx+hsx], # down-right
                      [cz, cy-hsy, cx-hsx]  # down-left
                      ]
        rectangles.append(rectangle)
    return np.array(rectangles)
        
    
@magicgui(call_button="Register ROIs",
          mode={"choices": ['Translation','Affine','Euclidean','Homography']})
def register_rois(image: Image,
                    initial_rois: Labels,
                    mode: str = 'Translation',
                    median_filter_size:int = 3,
                    scale = 0.5,
                    bbox_zoom = 2
                    ):
    '''
    Registers rectangular rois chosen on image as the bound box of the labels.
    Based on cv2 registration.
    '''
    print('Starting registration...')
    register_rois.enabled = False
    points_layer_name = f'centroids {image.name}'
    rectangles_name = f'rectangles {image.name}'
    # remove registration points if present
    label_values = max_projection(initial_rois)
    label_colors = get_labels_color(label_values)
    labels = max_projection(initial_rois)
    initial_time_index = viewer.dims.current_step[0]
    real_initial_positions, real_roi_sy, real_roi_sx = get_rois_props(labels, 
                                                                      initial_time_index,
                                                                      bbox_zoom) 
    roi_num = len(real_initial_positions)
    stack = np.asarray(image.data)
    time_frames_num, _, _ = stack.shape
    
    if points_layer_name in viewer.layers:
        viewer.layers.remove(points_layer_name)
    if rectangles_name in viewer.layers:
        viewer.layers.remove(rectangles_name)
        
    def add_rois(params):
        import numpy.matlib
        rectangles = params[0]
        centers = params[1]
        rectangles = rectangles.reshape((roi_num*time_frames_num,4,3))
        centers = centers.reshape((roi_num*time_frames_num,3))
        
        color_array= np.matlib.repmat(label_colors,len(rectangles)//roi_num,1)
        shapes = viewer.add_shapes(np.array(rectangles[0]),
                          edge_width=2,
                          edge_color=color_array[0],
                          face_color=[1,1,1,0],
                          name = rectangles_name
                          )
        shapes.add_rectangles(np.array(rectangles[1:]),
                              edge_color=color_array[1:])
        
        viewer.add_points(np.array(centers),
                              edge_color='green',
                              face_color=[1,1,1,0],
                              name = points_layer_name
                              )
        register_rois.enabled = True
        print('... ending registration.') 
        
      
    @thread_worker(connect={'returned':add_rois})
    def _register_rois():    
        warnings.filterwarnings('ignore')        
        resized = resize_stack(stack, scale)
        resized, _vmin, _vmax = normalize_stack(resized)
        image0 = resized[initial_time_index,...]
        
        initial_positions= rescale_position(real_initial_positions,scale)
        roi_sy = [int(ri*scale) for ri in real_roi_sy]
        roi_sx = [int(ri*scale) for ri in real_roi_sx]
        previous_rois = select_rois_from_image(image0, initial_positions, roi_sy,roi_sx)

        rectangles = np.zeros([time_frames_num,roi_num,4,3]) 
        centers = np.zeros([time_frames_num,roi_num,3])
        
        # register forwards
        next_positions = initial_positions.copy()
        for t_index in range(initial_time_index, time_frames_num, 1):
            real_next_positions = rescale_position(next_positions,1/scale)
            centers[t_index,:,:] = np.array(real_next_positions)
            rectangles[t_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
            next_rois = select_rois_from_image(resized[t_index,...], next_positions, roi_sy,roi_sx)
            dx, dy = align_with_registration(next_rois,previous_rois,
                                             median_filter_size,
                                             mode)
            next_positions = update_position(next_positions, dz = 1,
                                             dx_list = dx, dy_list = dy)
        # register backwards  
        next_positions = initial_positions.copy()    
        for t_index in range(initial_time_index-1, -1, -1):
            next_rois = select_rois_from_image(resized[t_index,...], next_positions, roi_sy,roi_sx)
            dx, dy = align_with_registration(next_rois,previous_rois,
                                             median_filter_size,
                                             mode)
            next_positions = update_position(next_positions, dz = -1,
                                             dx_list = dx, dy_list = dy)
            real_next_positions = rescale_position(next_positions,1/scale)
            centers[t_index,:] = np.array(real_next_positions)
            rectangles[t_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
            
        return (rectangles, centers)
            
    _register_rois()
    
    
    

def calculate_intensity(image:Image,
                        roi_num:int,
                        points_layer:Points,
                        labels_layer:Labels,
                        ):
    """
    Calculates the mean intensity,
    within rectangular Rois of size roi_size, centered in points_layer,
    taking into account only the pixels that are in one of the labels of labels_layer
    """
    
    labels_data = max_projection(labels_layer)
    label_values = get_labels_values(labels_data)
    stack = image.data
    locations = points_layer.data
    st, _sy, _sx = stack.shape
    _ , roi_sizey, roi_sizex = get_rois_props(labels_data)   
    rois = select_rois_from_stack(stack, locations, roi_sizey, roi_sizex)
    label_rois = select_rois_from_image(labels_data, locations[0:roi_num], roi_sizey,roi_sizex)
    
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
                 initial_rois: Labels,
                 correct_photobleaching: bool,
                 plot_results:bool = True,
                 save_results:bool = False,
                 path: pathlib.Path = os.getcwd()+"\\test.xlsx",
                 ):
    warnings.filterwarnings('ignore')
    print('Starting processing ...')
    try:
        process_rois.enabled = False
        time_frames_num, sy, sx = image.data.shape
        locations = registered_points.data
        roi_num = len(locations) // time_frames_num
        intensities = calculate_intensity(image, roi_num, 
                                          registered_points,
                                          initial_rois)
        yx, deltar, dyx, dr = measure_displacement(image, roi_num, registered_points)
        
        if correct_photobleaching:
            intensities = correct_decay(intensities)
        spectra = calculate_spectrum(intensities)    
            
        if plot_results:
            label_values = max_projection(initial_rois)
            colors = get_labels_color(label_values)
            plot_data(deltar, colors, "time index", "lenght (px)")
            plot_data(intensities, colors, "time index", "mean intensity")
            #plot_data(spectra, colors, "frequency index", "power spectrum", plot_type = 'log')
        
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
    finally: process_rois.enabled = True
    print(f'... processed {time_frames_num} frames.')       
    
    
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    
    folder = os.getcwd() + "\\images"
    
    viewer.open(folder)
    viewer.layers[0].name = 'images'
   
    # add some labels
    _image = viewer.layers['images'].data
    st,sy,sx = _image.shape
    
    test_label = np.zeros([sy,sx],dtype=int)
    test_label[1029:1180, 801:870] = 1
    test_label[1320:1470, 600:670] = 7
   
    _labels_layer = viewer.add_labels(test_label, name='labels')

    # viewer.window.add_dock_widget(optimize_contrast, name = 'Set contrast',
    #                               area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(subtract_background, name = 'Subtract background',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(register_stack, name = 'Stack Registration',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(register_rois, name = 'ROIs Registration',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(process_rois, name = 'Processing',
                                  area='right')
    warnings.filterwarnings('ignore')
    
    napari.run() 
    
    
    
    
    