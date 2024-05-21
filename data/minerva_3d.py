# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
    Generate a 3D matrix of capacitence measurements    
    The matrix is of size (M, N, C): 

    Assume we are looking at a grid of electrodes fo size mxn, m rows and n columns: 
        - Row Offset Matrices (Row Shift), col slice: 
            - In this setup, we take a slice through each column in the grid, measure the capacitence along this column at varying spatial offsets.
            - We get a matrix of capacitence measurements of size (m x r) (W, H), 
               where r is the number of spatial offsets considered whwen measuring the capacitence between the rows and 
               m is the number of electrodes in the column slice which is equal to the number of rows
            - In total we get (W, H, C)=(m, r, n) 3-D matrix of capacitence measurements.    

        - Column Offset Matrices (Column Shift), row slice: 
            - In this setup, we take a lice through each row in the grid, measure the capacitence along this row at varyin spatial offsets. 
            - We get a matrix of capacitence measurements of size (rxn),
              where r is the number of spatial offsets considered whwen measuring the capacitence between the rows and 
              n is the number of electrodes in the row slice which is equal to the number of rows. 
            - In total we get (W, H, C)=(n, r, m) 3-D matrix of capacitence measurements.    

        - Diagonal Offset Matrices: 

"""

import os
import argparse 

import cv2
import numpy as np 
from matplotlib import pyplot as plt
from datetime import datetime

try:
    from utils import read_yaml, snr, estimate_noise
    from plot import plot, plot_image_slice, plot_slice, draw_ect, draw_line, plot_capacitence_3d
    from plot import interpolate_perm, plot_box_annotations
    from mesh_params import MeshParams
    from tree_generator import TreeGenerator
    from generate_data import ECTInstance, create_mesh
    from random_perm import set_perm
except Exception as e:
    from .utils import read_yaml, snr, estimate_noise
    from .plot import plot, plot_image_slice, plot_slice, draw_ect, draw_line, plot_capacitence_3d
    from .plot import interpolate_perm, plot_box_annotations
    from .mesh_params import MeshParams
    from .tree_generator import TreeGenerator
    from .generate_data import ECTInstance, create_mesh
    from .random_perm import set_perm


def plot_capacitence(image, slice_col, capacitence_3d, row_offset, col_offset, row_range, col_range, output_path):
    # plot the image slice 
    save_path = os.path.join(output_path, f"{slice_col}__row_{row_offset}_col_{col_offset}_box.png")
    plot_image_slice(image, slice_col, row_range, col_range, "Box annotations", save_path)

    save_path = os.path.join(output_path, f"{slice_col}__row_{row_offset}_col_{col_offset}_slice.png")
    plot_slice(image, slice_col, row_range, col_range, f"Image Slice, Row {row_offset}, Col {col_offset}", save_path)

    save_path = os.path.join(output_path, f"{slice_col}__row_{row_offset}_col_{col_offset}_ect_reading.png")
    plot_capacitence_3d(np.array(capacitence_3d),f"Capacitence Measurements at Row Offset {row_offset}, Col Offset {col_offset}", "", "", save_path)


def measure_noise(ground_reference, image):
    psnr = cv2.PSNR(ground_reference, image)
    return psnr 

def flatten(cap_meas, axis=-1, offset=1, drop_zeros=True):
    if axis == -1: 
        cap_meas_flattened = [cap_meas[:, :, i].flatten() for i in range(0, cap_meas.shape[-1])]
    elif axis == 1:
        cap_meas_flattened = [cap_meas[:, i, :].flatten() for i in range(0, cap_meas.shape[1])]
    
    if drop_zeros: 
        cap_meas_flattened_new = []
        spatial_offset = cap_meas.shape[0]

        indices = []
        for i in range(1, spatial_offset+1):
            for k in range(1, i+offset):
                indices.append(cap_meas.shape[1]*i-(k))

        for i in range(0, len(cap_meas_flattened)):
            new_x = np.delete(cap_meas_flattened[i], indices)
            cap_meas_flattened_new.append(new_x)
        
        return np.stack(cap_meas_flattened_new, axis=1)
    
    return np.stack(cap_meas_flattened, axis=1)


