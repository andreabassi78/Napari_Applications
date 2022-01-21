# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:54:28 2022

@author: andrea
"""

import numpy as np
from skimage import data
import napari

# create "time point"
# (z stack with 3 color channels, shape 3x128x128x128)
blobs_t1 = np.stack(
        [
            data.binary_blobs(
                length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
            )
            for f in [0.05, 0.1, 0.15]
        ],
        axis=0,
)

# create second "time point"
# (z stack with 3 color channels, shape 3x128x128x128)
blobs_t2 = np.stack(
    [
        data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
        )
        for f in [0.20, 0.25, 0.30]
    ],
    axis=0,
)

# combine time points
# 2 x 3 x 128 x 128 x 128 stack
blobs = np.stack([blobs_t1, blobs_t2])
print(blobs_t1.shape)
print(blobs_t2.shape)
print(blobs.shape)

viewer = napari.Viewer()
viewer.add_image(blobs.astype(float), channel_axis=1, name=['dapi', 'gfp', 'rfp'])
#viewer.add_image(blobs.astype(float))

