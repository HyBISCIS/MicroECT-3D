# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os
import sys
import cv2
import glob
import json 
import copy 
import random 
random.seed(0)

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as T

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, median_filter

try:
    from plot import draw_grid, draw_line, draw_boundary_measurement, plot_3d_volume
    from dataset import xData, yData
    from minerva_3d import flatten
    from capacitence import CAP, CAPMEAS
except:
    from .plot import draw_grid, draw_line, draw_boundary_measurement, plot_3d_volume
    from .dataset import xData, yData
    from .minerva_3d import flatten
    from .capacitence import CAP, CAPMEAS


class Dataset3D(torch.utils.data.IterableDataset):

    def __init__(self, path, shuffle=False, normalize=False, standardize=False, global_scaling=False, smooth=False, drop_zeros=False, pos_value=-1, neg_value=1, train_min=None, train_max=None, cap_meas=[CAP.COLUMN_SHIFT], device='cuda'):
        if not os.path.exists(path):
            print(f"{path} doesn't exist.")
            sys.exit(1)

        self.path = path
        self.shuffle = shuffle 
        self.normalize = normalize  
        self.global_scaling = global_scaling
        self.standardize = standardize
        self.smooth = smooth
        self.drop_zeros = drop_zeros
        self.device = device 
        self.pos_value = pos_value
        self.neg_value = neg_value
        self.train_min = train_min
        self.train_max = train_max 
        self.cap_meas_type = cap_meas 
        
        self.json_path = os.path.join(self.path, f"dataset.json")
        
        if self.standardize or self.normalize: 
            self.vb_mean, self.vb_std, self.vb_min, self.vb_max = self.calc_vb_stats()
            self.img_mean, self.img_std, self.img_min, self.img_max = self.calc_image_stats()

        self.xData = []
        self.yData = []
        self.parse_data()

    def calc_image_stats(self):
        # Get list of all images in training directory
        perm_path = os.path.join(os.path.join(self.path, f"x/perm_xy/*.npy"))
        file_list = sorted(glob.glob(str(perm_path)))

        images = []
        for file in file_list:
            img = np.load(file)
            images.append(img)

        mean = np.mean(images, axis=0)
        std = np.std(images, axis=0)

        min = np.min(images)
        max = np.max(images)

        return mean, std, min, max

    def calc_vb_stats(self):
        # Get list of all images in training directory
        vb_path = os.path.join(os.path.join(self.path, f"y/v_b/*.npy"))
        file_list = sorted(glob.glob(vb_path))
        
        cap_meas = CAPMEAS()
        for file in file_list:
            with open(file, 'rb') as f:
                column_shift_data, row_shift_data, diagonal_shift_data_1, diagonal_shift_data_2 = pickle.load(f)
                
                row_shift_data = row_shift_data.reshape(column_shift_data.shape)
                diagonal_shift_data_1 = diagonal_shift_data_1.reshape(column_shift_data.shape)
                diagonal_shift_data_2 = diagonal_shift_data_2.reshape(column_shift_data.shape)
                # cap_all = np.concatenate((column_slice_data, row_slice_data, diagonal_slice_data_1, diagonal_slice_data_2), axis=2)
            
                cap_meas.row_shift.append(row_shift_data)
                cap_meas.column_shift.append(column_shift_data)
                cap_meas.diag_shift_1.append(diagonal_shift_data_1)
                cap_meas.diag_shift_2.append(diagonal_shift_data_2)

        if self.global_scaling: 
            mean = cap_meas.global_mean
            std = cap_meas.global_std
            min = cap_meas.global_min
            max = cap_meas.global_max
        else: 
            mean = cap_meas.mean
            std = cap_meas.std
            min = cap_meas.min
            max = cap_meas.max

        if self.train_min is None: 
            self.train_min = min
            
        if self.train_max is None: 
            self.train_max = max

        print(self.train_min)
        print(self.train_max)
        
        return mean, std, self.train_min, self.train_max 


    def gaussian(input, is_training, stddev=0.2):
        if is_training:
            return input + Variable(torch.randn(input.size()).cuda() * stddev)
        return input


    def preprocess(self, img):
        # preprocess cross-sectional image
        img_normalized = copy.copy(img)
        img_normalized[img == 1] = self.neg_value
        img_normalized[img != 1] = self.pos_value
        
        return img_normalized 

    def preprocess_cap(self, cap, cap_type):
        # preprocess capacitence measurements
        vb_normalized  = cap

        if not self.global_scaling: 
            if self.normalize : 
                vb_normalized =  (cap - self.vb_min[cap_type]) / (self.vb_max[cap_type] - self.vb_min[cap_type])
                
            if self.standardize: 
                vb_normalized = cap - self.vb_mean[cap_type] / self.vb_std[cap_type] 
        else: 
            if self.normalize : 
                vb_normalized =  (cap - self.vb_min) / (self.vb_max - self.vb_min)
                
            if self.standardize: 
                vb_normalized = cap - self.vb_mean / self.vb_std
                
        return vb_normalized 
    
    def parse_data(self):
        self.perm = []
        self.capacitence = []
        self.ext_mat = []
        self.number_lines = []
        self.num_beads = []
        
        f = open(self.json_path)
        data = json.load(f)

        for _, subdict in data.items():
            perm_path = os.path.join(self.path, subdict["perm"])
            perm_xy_path = os.path.join(self.path, subdict["perm_xy"]) 
            dperm_path = os.path.join(self.path, subdict["dperm"]) 

            xdata = xData(perm_path, perm_xy_path , dperm_path)
            self.xData.append(xdata)

            num_beads = subdict["num_beads"]
            self.num_beads.append(num_beads) 
            
            v_b = os.path.join(self.path, subdict["v_b"]) 

            # read FEM solutions
            solutions = subdict["u"]

            y = yData(v_b)
            for solution in solutions.items():
                _, solution = solution
                ext_elec = solution["ext_elec"]
                ext_elec_pos = solution["ext_elec_pos"]
                u = os.path.join(self.path, solution["u"])  
                u_xy = os.path.join(self.path, solution["u_xy"]) 
                du = os.path.join(self.path, solution["du"])  

                y.add_solution(u, u_xy, du, ext_elec, ext_elec_pos)

            self.yData.append(y)

        if self.shuffle: 
            c = list(zip(self.xData, self.yData))
            random.shuffle(c)
            self.xData, self.yData = zip(*c)

        self.num_datapoints = len(self.xData)

    def __iter__(self):
        for xdata, ydata in zip(self.xData, self.yData):
            x = xdata.load()
            y = ydata.load(is_pickle=True)
            x, y = self.load_x_y(x, y)
            yield x, y
    
    def smooth_edges(self, perm_xy, sigma=1):
        perm_smoothed = median_filter(perm_xy, size=5)
        return perm_smoothed
    
    def __getitem__(self, idx):
        xdata = self.xData[idx]
        ydata = self.yData[idx]

        x = xdata.load()
        y = ydata.load(is_pickle=True)
        
        x,y = self.load_x_y(x, y)

        return x, y 

    def load_x_y(self, x, y): 
          
        column_shift, row_shift, diagonal_shift_1, diagonal_shift_2 = y["v_b"]
        cap_meas = CAPMEAS(row_shift=row_shift, column_shift=column_shift, diag_shift_1=diagonal_shift_1, diag_shift_2=diagonal_shift_2)

        cap_all = []
        for cap_type in self.cap_meas_type:
            cap = cap_meas.get(cap_type)
            
            offset = 1
            if cap_type in [CAP.DIAGONAL_SHIFT_1.value, CAP.DIAGONAL_SHIFT_2.value]:
                offset = 2
            
            axis = -1
            if cap_type in [CAP.COLUMN_SHIFT.value, CAP.DIAGONAL_SHIFT_2.value]:
                axis = 1 
            
            cap_flattened = flatten(cap, axis=axis, offset=offset, drop_zeros=self.drop_zeros)
            
            cap_flattened = torch.tensor(cap_flattened)
            
            cap_flattened = self.preprocess_cap(cap_flattened, cap_type)

            cap_all.append(cap_flattened)
        
        y["v_b"] = torch.cat(cap_all, dim=0)
        
        # preprocess 
        x["perm_xy"] = self.preprocess(x["perm_xy"])
        
        if self.smooth: 
            x["perm_xy"] = self.smooth_edges(x["perm_xy"])
        
        x["perm_xy"] = x["perm_xy"][:, 0:100, 0:200]
        return x, y
    
    
    def __len__(self):
        return self.num_datapoints


def main(): 
    batch_size = 1
   
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)

    root_dir = "logs"
    data_path = os.path.join(root_dir, "3D-Datasets/07082022_conf_ect_sweep_3d_fixed/dataset")

    dataset = Dataset3D(data_path, shuffle=False, normalize=True, smooth=True, standardize=False, drop_zeros=False, global_scaling=False, cap_meas=[CAP.ROW_SHIFT.value, CAP.COLUMN_SHIFT.value, CAP.DIAGONAL_SHIFT_1.value, CAP.DIAGONAL_SHIFT_2.value])
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    for x, y in data_loader:
        
        print("Dataset")
        # 3-D confocal volme
        volume = x["perm_xy"][0]
        print("3-D Shape: ", volume.shape)
        plot_3d_volume(volume, 0, debug_dir)

        # 3-D capacitence values 
        print(y["v_b"])
        print(y["v_b"].shape)

        col_slice_meas = y["v_b"][0]
        print("Capacitence Measurements Shape (Column Slice): ", col_slice_meas.shape)
        
        # plot the capacitence measurements for each column slice
        for col in range(0, col_slice_meas.shape[1]): 
            cap_meas = col_slice_meas[:, col]
            print(cap_meas.shape)
            draw_line(np.arange(0, len(cap_meas)),cap_meas, "Boundary Measurement", "Measurement", "Capacitence Value", os.path.join(debug_dir, f"{col}_vb.png"))

        break 


if __name__ == "__main__":
    main()