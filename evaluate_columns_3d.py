# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os 
import time
import math
import argparse 
import cv2 
import tifffile
import numpy as np
import torch
import cv2 

from torch.utils.data import DataLoader

from data.dataset import Dataset
from data.plot import draw_grid, draw_confocal_grid, sweep_frame

from GAN.model_3d import Generator3d, ResidualGenerator3d, ResidualGenerator3d_2
from GAN.train import test, smooth_predictions
from GAN.train_3d import scale_volume_x
from GAN.loss import get_loss 

from scipy.ndimage import gaussian_filter, median_filter

from data.utils import read_yaml, resize_cfg
from data.confocal import read_confocal, build_depth_image_2, conf_image_size, preprocess, plot_conf_images, plot_confocal, fix_column_tilt
from data.minerva import read_ect
from data.minerva_3d import get_ect_3d_2, flatten
from data.capacitence import CAP, CAPMEAS
from data.confocal_3d import build_3d_volume

from config import combine_cfgs
from utils import init_torch_seeds, load_checkpoint
from experiments.tree_generator import TreeGenerator
from metrics.metrics import Metrics, tabulate_runs


def post_process(pred):
    # 2. Remove small dots from image 
    # kernel = np.ones((10, 10),np.uint8)
    # img_dilation = cv2.dilate(pred, kernel, iterations=1)
    img_dilation = median_filter(pred, size=7)
    kernel = np.ones((3, 3),np.uint8)
    erosion = cv2.erode(img_dilation, kernel, iterations = 1)
    # fill 
    return erosion 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=True)
    parser.add_argument('--model', type=str, help="Path to the trained model", required=False)
    parser.add_argument('--ect', type=str, help="Path to ECT Data", required=True)
    parser.add_argument('--confocal', type=str, help="Path to Confocal Data", required=True)
    parser.add_argument('--ect_cfg', type=str, help="Path to ECT Data", required=True)
    parser.add_argument('--confocal_cfg', type=str, help="Path to Confocal Data", required=True)
    parser.add_argument('--slice_col', type=int, help="Column Slice to Predict", required=True)
    parser.add_argument('--batch_size', type=int, help="Batch Size", required=False, default=1)
    parser.add_argument('--output_dir', type=str, help="Batch Size", required=False, default="logs/column2")

    args = parser.parse_args()
    
    model_path = args.model 
    ect_file = args.ect 
    confocal_file = args.confocal
    ect_cfg_file = args.ect_cfg
    confocal_cfg_file = args.confocal_cfg
    slice_col = args.slice_col
    output_dir = args.output_dir 

    config = combine_cfgs(args.config)

    seed = config.SEED 
    exp_name = config.NAME 
    num_measurements = config.DATASET.NUM_MEASUREMENTS 
    head_activation = config.MODEL.HEAD_ACTIVATION
    hidden_activation = config.MODEL.HIDDEN_ACTIVATION 
    loss = config.SOLVER.LOSS
    model_type = config.MODEL.TYPE
    cap_meas_type = config.DATASET.CAP_MEAS
    drop_zeros = config.DATASET.DROP_ZEROS
    test_min = config.DATASET.TEST_MIN
    test_max = config.DATASET.TEST_MAX
    
    if args.batch_size: 
        batch_size = args.batch_size 
    else: 
        batch_size = config.DATASET.BATCH_SIZE
   
    # save_path = os.path.join('experiments', exp_name)
    # output_dir = os.path.join(save_path, "eval")

    if model_path is None: 
        model_path = os.path.join(save_path, 'best_model.pth')
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_tree = TreeGenerator(root_dir=output_dir)
    output_tree.generate()
    
    # Prepare model and load parameters
    if model_type == 'Vanilla-Decoder':
        model = Generator3d(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)
    else: 
        model = ResidualGenerator3d_2(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)

    model.load_state_dict(torch.load(model_path)['state_dict']) 
    model = model.to(device)


    # Read ECT & Confocal Datasets
    ect_cfg = read_yaml(ect_cfg_file)
    conf_cfg = read_yaml(confocal_cfg_file) 
    
    conf_img_stack, conf_image, conf_maxZ = read_confocal(confocal_file, conf_cfg, output_dir)
    print("Confocal shape: ", conf_image.shape, flush=True)
    
    tifffile.imwrite(os.path.join(output_dir, "confocal.tif"), np.array(conf_img_stack).astype(np.float64), bigtiff=True)

    # Preprocess confocal data 
    _, x, z = conf_img_stack.shape
    columns = []
    for i in range(0, x): 
        column_cross_section = [conf_img_stack[:, i, z-j-1] for j in range(1, z+1)]
        column_cross_section = np.array(column_cross_section).reshape((z, -1))
        if conf_cfg.FIX_TILT: 
            column_cross_section = fix_column_tilt(column_cross_section, conf_cfg)

        columns.append(column_cross_section)

    columns = np.array(columns)
    columns_processed = []
    column_fluoresence = []
    for i in range(0, columns.shape[0]): 
        col_processed = preprocess(columns[i, :, :], conf_cfg, os.path.join(output_dir,  f"{i}_cross_section_processed.png"))
        col_processed[col_processed == 255] = conf_cfg.FOREGROUND_PERM
        col_processed[col_processed == 0] = conf_cfg.BACKGROUND_PERM
        
        columns_processed.append(col_processed)
        column_fluoresence.append(columns[i, :, :])
        
        save_path = os.path.join(output_dir, f"column_{i}.png")
        plot_confocal(columns[i, :, :], "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, save_path=save_path, aspect_ratio=4.5,figsize=(12,12))

        save_path = os.path.join(output_dir,  f"column_processed_{i}.png")
        plot_confocal(col_processed, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, save_path=save_path, aspect_ratio=4.5,figsize=(12,12))


    ect_images, row_offsets, col_offsets = read_ect(ect_file, ect_cfg, output_dir)
    print("Row Offsets: ", row_offsets, flush=True)
    
    num_rows = ect_images[0].shape[0] - ect_cfg.ROW_OFFSET
    stride = ect_cfg.ROW_OFFSET
    
    predictions = torch.tensor([], device=device)
    pred_processed = torch.tensor([], device=device)
    ground_truth = torch.tensor([], device=device, dtype=torch.float32)
    ground_truth_processed = torch.tensor([], device=device, dtype=torch.float32)

    min = {"ROW": test_min[0], "COL": test_min[1], "DIAG_1": test_min[2], "DIAG_2": test_min[3]}
    max = {"ROW": test_max[0], "COL": test_max[1], "DIAG_1": test_max[2], "DIAG_2": test_max[3]}

    print(min, max, flush=True)

    # for i in range(100, 360, stride):
    for i in range(0, num_rows, stride):
        row_range = [i, i+ect_cfg.ROW_OFFSET]

        # get corresponding cross sectional image from  confocal 
        if not conf_cfg.RESIZE_STACK: 
            confocal_column = math.ceil((slice_col * 10) / conf_cfg.PIXEL_SIZE_XY)
            conf_step = math.ceil((i*10)/conf_cfg.PIXEL_SIZE_XY) + 1
            conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]
        else:
            confocal_column = slice_col
            conf_step = i*10
            conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]
            
        # quit if the confocal range is above the confocal image
        if conf_row_range[1] > conf_image.shape[0]: 
            break 
        
        minerva_data = get_ect_3d_2(ect_images, row_offsets, col_offsets, 
                                    ect_cfg.MAX_ROW_OFFSET, 
                                    subgrid_origin=(slice_col, i), 
                                    num_rows=ect_cfg.ROW_OFFSET,
                                    num_cols=ect_cfg.COL_OFFSET,
                                    config=ect_cfg,
                                    output_path=output_dir) 
        
        column_shift, row_shift, diagonal_shift_1, diagonal_shift_2 = minerva_data
        
        column_shift = column_shift*1e15*0.1 
        row_shift = row_shift* 1e15*0.1 
        diagonal_shift_1 = diagonal_shift_1* 1e15*0.1 
        diagonal_shift_2 = diagonal_shift_2* 1e15*0.1 
            
        # min max scaling for the data 
        column_shift = (column_shift - min["COL"]) / (max["COL"] - min["COL"])
        row_shift = (row_shift - min["ROW"]) / (max["ROW"] - min["ROW"])
        diagonal_shift_1 = (diagonal_shift_1 - min["DIAG_1"]) / (max["DIAG_1"] - min["DIAG_1"])
        diagonal_shift_2 = (diagonal_shift_2 - min["DIAG_2"]) / (max["DIAG_2"] - min["DIAG_2"])

        cap_meas = CAPMEAS(row_shift=row_shift, column_shift=column_shift, diag_shift_1=diagonal_shift_1, diag_shift_2=diagonal_shift_2)

        cap_all = []
        for cap_type in cap_meas_type:
            cap = cap_meas.get(cap_type)
            
            offset = 1
            if cap_type in [CAP.DIAGONAL_SHIFT_1.value, CAP.DIAGONAL_SHIFT_2.value]:
                offset = 2
            
            axis = -1
            if cap_type in [CAP.COLUMN_SHIFT.value, CAP.DIAGONAL_SHIFT_2.value]:
                axis = 1 

            cap_flattened = flatten(cap, axis=axis, offset=offset, drop_zeros=drop_zeros)
            
            cap_flattened = torch.tensor(cap_flattened)
            
            cap_all.append(cap_flattened)
        
        vb = torch.cat(cap_all, dim=0)
        
        vb = vb.view(1, vb.shape[0], vb.shape[1], 1, 1)
        vb = vb.to(device)
        
        predicted_perm = model(vb.float())
        pred_perm_smoothed, _ = smooth_predictions(predicted_perm, torch.tensor([]), config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)
        # pred_perm_smoothed = predicted_perm
        
        predictions = torch.cat((predictions, pred_perm_smoothed), 4)
        
        pred_processed_pred = post_process(pred_perm_smoothed[0][0].cpu().detach().numpy()).reshape(1, 1, 5, 100, 200)
        print(pred_processed_pred.shape)

        pred_processed = torch.cat((pred_processed, torch.tensor(pred_processed_pred, device=device)), 4)
        print(pred_processed.shape)

        print("predictions: ", predictions.shape, flush=True)
        save_path =  os.path.join(output_tree.pred_path, f"pred_start_row_{i}.tif")
        tifffile.imwrite(save_path, pred_perm_smoothed[0][0].detach().cpu().numpy().astype(np.float32), compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

        save_path =  os.path.join(output_tree.pred_path, f"pred_start_row_{i}.png")
        draw_grid(np.max(pred_perm_smoothed[0][0].detach().cpu().numpy(), axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", save_path, cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=1, figsize=(12,12), scale_bar_box_alpha= 1, scale_bar_text_color = 'k', font_size=12, format="png")

        # # draw_grid(pred_perm_smoothed[0][0].cpu().detach().numpy(), "predicted_perm", "", "", os.path.join(output_tree.pred_path, f"pred_{i}.png"))

        # pred_processed_pred = post_process(pred_perm_smoothed[0][0].cpu().detach().numpy()).reshape(1, 1, pred_perm_smoothed.shape[2], pred_perm_smoothed.shape[3])
        # pred_processed = torch.cat((pred_processed, torch.tensor(pred_processed_pred, device=device)), 3)

        column_fluoresence = np.array(column_fluoresence)
        sub_columns = column_fluoresence[slice_col:slice_col+ect_cfg.COL_OFFSET, :, :]
        cropped_3d_volume = build_3d_volume(sub_columns, row_range=conf_row_range) 
        cropped_3d_volume = np.array(cropped_3d_volume, dtype=np.float32)

        columns_processed = np.array(columns_processed)
        sub_columns = columns_processed[slice_col:slice_col+ect_cfg.COL_OFFSET, :, :]

        cropped_3d_volume_processed = build_3d_volume(sub_columns, row_range=conf_row_range) 
        cropped_3d_volume_processed = np.array(cropped_3d_volume_processed, dtype=np.float32)

        print("ground truth: ", cropped_3d_volume.shape, flush=True)
        save_path = os.path.join(output_tree.true_path, f"ground_truth_start_row_{i}.tif")
        tifffile.imwrite(save_path, cropped_3d_volume,  compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

        save_path =  os.path.join(output_tree.true_path, f"ground_truth_start_row_{i}.png")
        draw_grid(np.max(cropped_3d_volume, axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", save_path, cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=1, figsize=(12,12), scale_bar_box_alpha= 1, scale_bar_text_color = 'k', font_size=12, format="png")

        save_path = os.path.join(output_tree.true_path, f"ground_truth_mask_start_row_{i}.tif")
        tifffile.imwrite(save_path, cropped_3d_volume_processed,  compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

        save_path =  os.path.join(output_tree.true_path, f"ground_truth_mask_start_row_{i}.png")
        draw_grid(np.max(cropped_3d_volume_processed, axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", save_path, cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=1, figsize=(12,12), scale_bar_box_alpha= 1, scale_bar_text_color = 'k', font_size=12, format="png")


        cropped_3d_volume = torch.tensor(cropped_3d_volume, device=device)        
        ground_truth = torch.cat((ground_truth, cropped_3d_volume), 2)

        cropped_3d_volume_processed = torch.tensor(cropped_3d_volume_processed, device=device)        
        ground_truth_processed = torch.cat((ground_truth_processed, cropped_3d_volume_processed), 2)
        print("ground truth: ", ground_truth.shape, flush=True)


    # flatten the predictions
    predictions = predictions[0][0].cpu().detach().numpy()
    pred_processed = pred_processed[0][0].cpu().detach().numpy()
    ground_truth = ground_truth.cpu().detach().numpy()
    ground_truth_processed = ground_truth_processed.cpu().detach().numpy()

    predictions_scaled = scale_volume_x(predictions, dsize=(50, 100, predictions.shape[-1]))
    pred_processed_scaled = scale_volume_x(pred_processed, dsize=(50, 100, predictions.shape[-1]))

    print("prediction scaled: ", predictions_scaled.shape)
    ground_truth_scaled = scale_volume_x(ground_truth, dsize=(50, 100, ground_truth.shape[-1]))

    # predictions_scaled = cv2.cvtColor(predictions_scaled, cv2.COLOR_GRAY2RGB)
    # predictions = cv2.resize(predictions, None, fx=1, fy=1)
    # ground_truth = cv2.resize(ground_truth, None, fx=1, fy=1)
    
    save_path =  os.path.join(output_tree.pred_path, f"pred_scaled_{slice_col}.tif")
    tifffile.imwrite(save_path, predictions_scaled,  compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

    save_path = os.path.join(output_tree.pred_path, f"pred_processed_{slice_col}.tif")
    tifffile.imwrite(save_path, pred_processed,  compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

    save_path = os.path.join(output_tree.pred_path, f"pred_processed_scaled_{slice_col}.tif")
    tifffile.imwrite(save_path, pred_processed_scaled,  compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)


    save_path =  os.path.join(output_tree.pred_path, f"pred_{slice_col}.tif")
    tifffile.imwrite(save_path, predictions, compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)
    
    save_path =  os.path.join(output_tree.true_path, f"truth_scaled_{slice_col}.tif")
    tifffile.imwrite(save_path, ground_truth_scaled, compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

    save_path =  os.path.join(output_tree.true_path, f"truth_{slice_col}.tif")
    tifffile.imwrite(save_path, ground_truth, compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

    save_path =  os.path.join(output_tree.true_path, f"truth_mask_{slice_col}.tif")
    tifffile.imwrite(save_path, ground_truth_processed, compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

    print("Final prediction: ", predictions.shape, flush=True)
    print("Final prediction: ", ground_truth.shape, flush=True)

    # draw the columns flattened
    for i in range(0, predictions.shape[0]):
        draw_grid(predictions[i, :, :],  f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"pred_{i}_{slice_col}.png"),  figsize=(26, 13), cmap='viridis', colorbar=False, scale_bar=True, ticks=False, aspect_ratio=5.5, font_size=12)

        draw_grid(predictions[i, :, :],  f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"pred_{i}_{slice_col}.pdf"),  figsize=(26, 13), cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=5.5, font_size=12, format="pdf")
        draw_grid(ground_truth[i, :, :], f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"truth_{i}_{slice_col}.png"),  figsize=(26, 13), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=5.5, font_size=12)
        draw_grid(ground_truth[i, :, :], f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"truth_{i}_{slice_col}.pdf"),  figsize=(26, 13), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=5.5, font_size=12, format="pdf")

    # draw the max projections
    draw_grid(np.max(predictions, axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.pred_path, f"pred_max_{slice_col}.png"), cmap='viridis', colorbar=False, scale_bar=True, ticks=False, aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1, scale_bar_length_fraction=0.2, scale_bar_text_color = 'k', font_size=20)
    draw_grid(np.max(predictions, axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.pred_path, f"pred_max_{slice_col}.pdf"), cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1,  scale_bar_length_fraction=0.2, scale_bar_text_color = 'k', font_size=20, format="pdf")
    
    draw_grid(np.max(pred_processed, axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.pred_path, f"pred_max_proc_{slice_col}.png"), cmap='viridis', colorbar=False, scale_bar=True, ticks=False, aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1, scale_bar_length_fraction=0.2, scale_bar_text_color = 'k', font_size=20)
    draw_grid(np.max(pred_processed, axis=0), f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.pred_path, f"pred_max_proc_{slice_col}.pdf"), cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1,  scale_bar_length_fraction=0.2, scale_bar_text_color = 'k', font_size=20, format="pdf")
    
    draw_grid(np.max(ground_truth, axis=0), f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.true_path, f"truth_max_{slice_col}.png"), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1,  scale_bar_length_fraction=0.2, scale_bar_text_color = 'k',font_size=20)
    draw_grid(np.max(ground_truth_processed, axis=0), f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.true_path, f"truth_max_mask_{slice_col}.png"), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1,  scale_bar_length_fraction=0.2, scale_bar_text_color = 'k',font_size=20)

    draw_grid(np.max(ground_truth, axis=0), f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.true_path, f"truth_max_{slice_col}.pdf"), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=3, figsize=(12,12), scale_bar_box_alpha= 1,  scale_bar_length_fraction=0.2, scale_bar_text_color = 'k', font_size=20, format="pdf")

    # predictions_flattened = torch.flatten(predictions, 0, 1)
    # ground_truth_flattened = torch.flatten(ground_truth, 0, 1)
    
    # print(predictions_flattened.shape)
    # print(ground_truth_flattened.shape)
    
    # metrics = Metrics(device=device)
    # metrics = metrics.forward(predictions, ground_truth)
    # print(metrics)

    # save_path = os.path.join(output_dir, "stats.json")
    # stats, table = tabulate_runs([metrics], run_time, save_path)
    # print(table.draw())


if __name__ == "__main__":
    main()