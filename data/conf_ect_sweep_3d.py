# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
    Generate 3-D datasets from the confocal and minerva data
"""

import sys
import os 
import math
import argparse 
import imageio
import cv2
import numpy as np 

try: 
    from utils import read_yaml, resize_cfg
    from confocal import read_confocal, fix_column_tilt, preprocess, plot_conf_images
    from confocal_3d import get_columns, build_3d_volume, plot_3d_volume
    from minerva import read_ect
    from minerva_3d import get_ect_3d_2
    from plot import plot_confocal_ECT, draw_line, draw_grid, plot, sweep_frame, sweep_frame_2, plot_confocal
    from minerva import simulate_biofilm
    from generate_data import ECTInstance
    from tree_generator import TreeGenerator
except Exception as e: 
    from .utils import read_yaml, resize_cfg
    from .confocal import read_confocal, fix_column_tilt, preprocess, plot_conf_images
    from .confocal_3d import get_columns, build_3d_volume, plot_3d_volume
    from .minerva import read_ect
    from .minerva_3d import get_ect_3d_2
    from .plot import plot_confocal_ECT, draw_line, draw_grid, plot, sweep_frame, sweep_frame_2, plot_confocal
    from .minerva import simulate_biofilm
    from .generate_data import ECTInstance
    from .tree_generator import TreeGenerator


index = 0 

def save_3d(minerva_data, cropped_3d_volume_flu, confocal_volume, tree_generator): 
    global index 
    # save to their corresponding dir 
    ect_instance = ECTInstance(index, num_beads=0, perm=[], perm_xy=confocal_volume, dperm=[], ext_mat=[])
    ect_instance.v_b = minerva_data
    ect_instance.write(tree_generator)

    # save_path = os.path.join(tree_generator.vb_dir, f"{index}__beads_{0}.png")
    # draw_line(np.arange(0, ect_instance.v_b.shape[0]), ect_instance.v_b, "Capacitence", "index", "Capacitence", save_path)
    # plot_3d_volume(confocal_volume, index, tree_generator.perm_xy_dir)
    
    # ect_instance.plot(mesh_points=None, mesh_triangles=None, mesh_params=None, el_pos=None, output_tree=tree_generator, debug=True)
    tree_generator.write_json({f"{index}": ect_instance.dict(tree_generator.root_dir, is_exp=True)})

    # save fluoresence volume 
    perm_file_name = f"{index}_flu.npy"
    with open( os.path.join(tree_generator.perm_xy_dir, perm_file_name), 'wb') as f:
        np.save(f, np.asarray(cropped_3d_volume_flu))
                
    index = index + 1 

def generate_data(slice_col, i, ect_images, row_offsets, col_offsets, column_fluoresence, columns_processed, conf_image, ect_cfg, conf_cfg, output_dir, tree_generator, save_all):
    y, x = columns_processed[0].shape
    
    column_shift_data, row_shift_data, diagonal_data_1, diagonal_data_2 = get_ect_3d_2(ect_images, row_offsets, col_offsets, ect_cfg.MAX_ROW_OFFSET, subgrid_origin=(slice_col, i), num_rows=ect_cfg.ROW_OFFSET, num_cols=ect_cfg.COL_OFFSET, config=ect_cfg, output_path=output_dir)
    
    # get corresponding cross sectional image from  confocal 
    if not conf_cfg.RESIZE_STACK: 
        confocal_column = math.ceil((slice_col * 10) / conf_cfg.PIXEL_SIZE_XY)
        conf_step = math.ceil((i*10)/conf_cfg.PIXEL_SIZE_XY) + 1
        conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]
    else:
        confocal_column = slice_col
        conf_step = i*10
        conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]


    columns_processed = np.array(columns_processed)
    column_fluoresence = np.array(column_fluoresence)

    sub_columns = columns_processed[slice_col:slice_col+ect_cfg.COL_OFFSET, :, :]
    sub_columns_flu = column_fluoresence[slice_col:slice_col+ect_cfg.COL_OFFSET, :, :]

    cropped_3d_volume = build_3d_volume(sub_columns, row_range=conf_row_range)
    cropped_3d_volume_flu = build_3d_volume(sub_columns_flu, row_range=conf_row_range)

    # print(cropped_3d_volume.shape)
    column_shift_data = column_shift_data*1e15*0.1 
    row_shift_data = row_shift_data* 1e15*0.1 
    diagonal_data_1 = diagonal_data_1* 1e15*0.1 
    diagonal_data_2 = diagonal_data_2* 1e15*0.1 

    save_path = None 
    if save_all:       
        save_path = os.path.join(output_dir, f"frame_{i}.png")
    
    # im = sweep_frame(slice_col, confocal_column, row_range, conf_row_range, ect_images[row_offsets.index(-1)], conf_image,  minerva_data[:, 2], scaled_data[:, 2], cross_section_processed, cross_section_processed, cross_section, save_path)
    
    save_3d((column_shift_data, row_shift_data, diagonal_data_1, diagonal_data_2), cropped_3d_volume_flu, cropped_3d_volume, tree_generator)
    
    # return im 


def sweep_3d(slice_col, stride, ect_images, row_offsets, col_offsets, column_fluoresence, columns_processed, conf_image, ect_cfg, conf_cfg, output_dir, tree_generator, save_all):
    
    myframes = []
    num_rows = ect_images[0].shape[0] - ect_cfg.ROW_OFFSET
    
    for i in range(0, num_rows, stride): 
        frame = generate_data(slice_col, i, ect_images, row_offsets, col_offsets, column_fluoresence, columns_processed, conf_image, ect_cfg, conf_cfg, output_dir, tree_generator, save_all)
        myframes.append(frame)

    print(f"Done {slice_col}", flush=True)
    print(index, flush=True)
    # create .mp4 video file
    # imageio.mimsave(os.path.join(output_dir, f'frames_{slice_col}_2.mp4'), myframes, fps=5)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ect', type=str, help="Path to ECT .h5 file.", required=True)
    parser.add_argument('--confocal', type=str, help="Path to confocal .tiff file.", required=True)
    parser.add_argument('--ect_cfg', type=str, help="Path to ECT config. .yaml file.", required=True)
    parser.add_argument('--confocal_cfg', type=str, help="Path to confocal config. .yaml file.", required=True)
    parser.add_argument('--stride', type=int, help="value of stride", default=4)
    parser.add_argument('--save_all', type=bool, help="Save all files, useful for debugging", default=False)
    parser.add_argument('--output_dir', type=str, help="Path output directorty. ", required=True)

    args = parser.parse_args()

    ect_file = args.ect 
    confocal_file = args.confocal
    ect_cfg_file = args.ect_cfg
    confocal_cfg_file = args.confocal_cfg
    stride = args.stride
    save_all = args.save_all
    output_dir = args.output_dir
    
    conf_dir = os.path.join(output_dir, "Confocal")
    ect_dir = os.path.join(output_dir, "ECT")
    dataset_dir = os.path.join(output_dir, "dataset")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ect_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    tree_generator = TreeGenerator(root_dir=dataset_dir)
    tree_generator.generate() 

    # read yaml cfg 
    ect_cfg = read_yaml(ect_cfg_file)
    confocal_cfg = read_yaml(confocal_cfg_file)

    ### 1. Read confocal  ###
    conf_img_stack, conf_image, conf_maxZ = read_confocal(confocal_file, confocal_cfg, output_dir)

    conf_image_resized = cv2.resize(conf_image, dsize=(256, 512))

    plot_conf_images(conf_image, conf_maxZ, conf_image_resized, confocal_cfg, conf_dir)

    print("Confocal shape: ", conf_image.shape, flush=True)

    ### 2. Read ECT ###
    ect_images, row_offsets, col_offsets = read_ect(ect_file, ect_cfg, ect_dir)
    print("Row Offsets: ", row_offsets, flush=True)
   
    ## Plot ect v.s confocal ##
    save_path = os.path.join(output_dir, "conf_ect.png")
    conf_cfg_resized = resize_cfg(confocal_cfg, conf_image.shape, (512, 256))
    plot_confocal_ECT(conf_image_resized, ect_images[row_offsets.index(-1)], ect_cfg, conf_cfg_resized, save_path)

    # Preprocess confocal data 
    _, x, z = conf_img_stack.shape
    columns = []
    for i in range(0, x): 
        column_cross_section = [conf_img_stack[:, i, z-j-1] for j in range(1, z+1)]
        column_cross_section = np.array(column_cross_section).reshape((z, -1))
        if confocal_cfg.FIX_TILT: 
            column_cross_section = fix_column_tilt(column_cross_section, confocal_cfg)
        columns.append(column_cross_section)

    columns = np.array(columns)
    columns_processed = []
    column_fluoresence = []
    for i in range(0, columns.shape[0]): 
        col_processed = preprocess(columns[i, :, :], confocal_cfg, os.path.join(output_dir,  f"{i}_cross_section_processed.png"))
        col_processed[col_processed == 255] = confocal_cfg.FOREGROUND_PERM
        col_processed[col_processed == 0] = confocal_cfg.BACKGROUND_PERM
        
        columns_processed.append(col_processed)
        column_fluoresence.append(columns[i, :, :])
        
        save_path = os.path.join(output_dir, f"column_{i}.png")
        plot_confocal(columns[i, :, :], "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, save_path=save_path, aspect_ratio=4.5,figsize=(12,12))

        save_path = os.path.join(output_dir,  f"column_processed_{i}.png")
        plot_confocal(col_processed, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, save_path=save_path, aspect_ratio=4.5,figsize=(12,12))

    ## 3. Sweep ECT and Confocal Image ##
    start_column = ect_cfg.START_COLUMN
    end_column = ect_cfg.END_COLUMN - ect_cfg.COL_OFFSET

    for slice_col in range(start_column, end_column, ect_cfg.COL_OFFSET):
        sweep_3d(slice_col, stride, ect_images, row_offsets, col_offsets, column_fluoresence, columns_processed, conf_image, ect_cfg, confocal_cfg, output_dir, tree_generator, save_all) 
        print(f"Done {slice_col}", flush=True)



if __name__ == "__main__":
    main()