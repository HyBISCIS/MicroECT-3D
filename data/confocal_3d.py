# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
    Generate cropped 3-D volumes from the confocal z-stack 
"""

import os 
import sys
import argparse 

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import io
from skimage.transform import resize

# from xvfbwrapper import Xvfb
# vdisplay = Xvfb(width=1920, height=1080)
# vdisplay.start()
# from pyvirtualdisplay import Display
# import os
# display = Display(visible=0, size=(1280, 1024))
# display.start()

# import mayavi.mlab as mlab
# import mayavi 
#mlab.init_notebook('itk')

# mlab.options.offscreen = True
#mayavi.engine.current_scene.scene.off_screen_rendering = True
# mlab.test_plot3d()
# mlab.savefig("test.png")

try: 
    from .plot import plot_confocal, plot_box_annotations, draw_line, draw_grid, plot_3d_volume
    from .utils import read_yaml 
    from .generate_data import ECTInstance, create_mesh, create_ex_mat
    from .minerva import simulate_biofilm
    from .confocal import read_confocal, get_column, get_depth_image, plot_conf_images, conf_image_size,preprocess
    from .mesh_params import MeshParams
except Exception as e: 
    from plot import plot_confocal, plot_box_annotations, draw_line, draw_grid, plot_3d_volume
    from utils import read_yaml 
    from generate_data import ECTInstance, create_mesh, create_ex_mat
    from minerva import simulate_biofilm
    from confocal import read_confocal, get_column, get_depth_image, plot_conf_images, conf_image_size,preprocess
    from mesh_params import MeshParams


def get_columns(img_stack, slice_col, column_offset):
    y, x, z = img_stack.shape
    
    column_range = [slice_col, slice_col+column_offset]
    if column_range[0] < 0: 
        column_range[0] = 0
        column_range[1] = slice_col + column_offset*2 
    if column_range[1] > x: 
        column_range[0] = slice_col - column_offset*2
        column_range[1] = x 
    
    # Build the column cross section from the z-stack
    column_3d_cross_section = [img_stack[:, column_range[0]:column_range[1], z-j-1] for j in range(1, z+1)]
    column_3d_cross_section = np.array(column_3d_cross_section)
    
    return column_3d_cross_section



def build_3d_volume(cross_section_3d, row_range):
    cropped_volume = cross_section_3d[:, :, row_range[0]:row_range[1]]
    return cropped_volume 

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--confocal', type=str, help="Path to merged .tif file.", required=False)
    parser.add_argument('--confocal_cfg', type=str, help="Path to config file .yaml file.", required=True)
    parser.add_argument('--simulate', type=bool, help="Simulate the biofilm in pyEIT.", default=False)
    parser.add_argument('--input_dir', type=str, help="Path to input directory of .tif series if merged file not available.", required=False)
    parser.add_argument('--output_dir', type=str, help="Path to dataset output directory.", required=True)

    args = parser.parse_args()

    input_file = args.confocal 
    input_dir = args.input_dir 
    yaml_cfg = args.confocal_cfg
    simulate = args.simulate
    output_dir = args.output_dir 

    # input_file = "data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/Confocal/CFP_Biofilm.tif"
    # # input_dir = 
    # yaml_cfg = "data/real/06132023_Confocal.yaml"
    # simulate = False
    # output_dir = "data/logs/06132023_confocal_3d"
    # os.makedirs(output_dir, exist_ok=True)

    if input_file is None and input_dir is None: 
        print("[ERROR]: Missing input_file/iput_dir arguments.")
        sys.exit(1)

    # Read YAML CFG file 
    config = read_yaml(yaml_cfg)

    # Read .tif files
    img_stack, conf_image, conf_maxZ = read_confocal(input_file, config, output_dir)
    image_size = conf_image_size(conf_image, config)
    
    print("confoca z-stack size : ", img_stack.shape)
   
    # Resize confocal image to align with ECT image 
    conf_image_resized = cv2.resize(conf_image, dsize=(256, 512))
    
    # Plot images
    plot_conf_images(conf_image, conf_maxZ, conf_image_resized, config, output_dir)

    y, _, z = img_stack.shape
    for i, slice_col in enumerate(config.COLUMNS):
        columns = get_columns(img_stack, slice_col, config.COL_OFFSET) 
        
        columns_processed = []
        for k in range(0, config.COL_OFFSET):
            column = columns[:, :, k]
            save_path = os.path.join(output_dir,  f"{i}_cross_section_processed_{slice_col}.png")
            column_processed = preprocess(column, config, save_path)
            column_processed[column_processed == 255] = config.FOREGROUND_PERM
            column_processed[column_processed == 0] = config.BACKGROUND_PERM
            columns_processed.append(column_processed)

            save_path = os.path.join(output_dir, f"{k}_column_{slice_col}.png")
            plot_confocal(column, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, save_path=save_path, aspect_ratio=4.5,figsize=(12,12))

            save_path = os.path.join(output_dir,  f"{k}_column_processed_{slice_col}.png")
            plot_confocal(column, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, save_path=save_path, aspect_ratio=4.5,figsize=(12,12))

        columns_processed = np.array(columns_processed)

        # build 3-d volume 
        cropped_3d_volume = build_3d_volume(columns_processed, row_range=config.ROWS[i])
        print(cropped_3d_volume.shape)
        plot_3d_volume(cropped_3d_volume, 0, output_dir)
        
   


if __name__ == "__main__":
    main()
