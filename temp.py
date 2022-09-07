from magicgui import magicgui
import datetime
import pathlib
import os
    
@magicgui(call_button="Process")
def select_data(
    filename_0 = pathlib.Path(),
    filename_1 = pathlib.Path(),
    filename_2 = pathlib.Path(),
    filename_3 = pathlib.Path(),
    filename_4 = pathlib.Path(),
    filename_5 = pathlib.Path(),
    filename_6 = pathlib.Path(),
    filename_7 = pathlib.Path(),
    filename_8 = pathlib.Path(),
    filename_9 = pathlib.Path(),
    rois = '0',
    saving_file_name = 'temp'
    ):
    '''
    

    Parameters
    ----------
    filename_0 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_1 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_2 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_3 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_4 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_5 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_6 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_7 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_8 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_9 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    rois : TYPE, optional
        ..... for example '0,1,5'.
    saving_file_name : TYPE, optional
        DESCRIPTION. The default is 'temp'.

    Returns
    -------
    None.

    '''
    
    
    rois_list = rois.split(',')
    #data = [[None]*len(rois_list)]
    data = []
    print(rois_list)
    
    for idx,filename in enumerate([filename_0,filename_1,
                                   filename_2,filename_3,
                                   filename_4,filename_5,
                                   filename_6,filename_7,
                                   filename_8,filename_9]):  
        if filename !=pathlib.Path(): 
            folder = filename.parent
            roi_data = []
            for roi_idx,roi in enumerate(rois_list):
                roi_data.append(filename.name)
            data.append(roi_data)
    
    new_data = [list(x) for x in data]        
    print(new_data) 
    
    #savename= folder + '\\' + saving_file_name + '.xlsx'

    
select_data.show()

    
