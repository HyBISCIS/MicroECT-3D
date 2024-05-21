

import math
import cv2
import yaml 
import numpy as np
from skimage.restoration import estimate_sigma
from scipy.signal import convolve2d
from skimage.color import rgb2gray

class YamlCFG(object):
      
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def add_attr(self, key, val):
        setattr(self, key, val)


def read_yaml(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    yaml_cfg = YamlCFG(data_loaded)

    COLUMNS = [] 
    ROWS = []
    POSITION = []
    GRID_OFFSET = []
    for box in data_loaded["BOXES"]:
        COLUMNS.append(box['COLUMN'])
        ROWS.append([box["ROW"], box["ROW"]+yaml_cfg.ROW_OFFSET])  
        if "X" in box: 
            POSITION.append([box["X"], box["Y"]])

        if "GRID_OFFSET" in box: 
            GRID_OFFSET.append(box["GRID_OFFSET"])
            
    COL_RANGE = [[col-yaml_cfg.COL_OFFSET, col+yaml_cfg.COL_OFFSET] for col in COLUMNS]

    yaml_cfg.add_attr("ROWS", ROWS)
    yaml_cfg.add_attr("COL_RANGE", COL_RANGE)
    yaml_cfg.add_attr("COLUMNS", COLUMNS)
    yaml_cfg.add_attr("POSITION", POSITION)
    yaml_cfg.add_attr("GRID_OFFSET", GRID_OFFSET)
    
    # set default attributtes
    if not hasattr(yaml_cfg, "ROTATE_Z_STACK_180"): 
        yaml_cfg.add_attr("ROTATE_Z_STACK_180", False)

    if not hasattr(yaml_cfg, "WRAP_TRANSFORM"): 
        yaml_cfg.add_attr("WRAP_TRANSFORM", False)

    if not hasattr(yaml_cfg, "FIX_TILT"):
        yaml_cfg.add_attr("FIX_TILT", False)
    
    # CV Filter Configurations 
    if not hasattr(yaml_cfg, "EROSION_ITERS"):
        yaml_cfg.add_attr("EROSION_ITERS", [1, 0])

    if not hasattr(yaml_cfg, "EROSION_KERNEL"):
        yaml_cfg.add_attr("EROSION_KERNEL", [[3, 3], [1, 1]])
   
    if not hasattr(yaml_cfg, "DILATION_ITERS"):
        yaml_cfg.add_attr("DILATION_ITERS", 1)
    
    if not hasattr(yaml_cfg, "DILATION_KERNEL"):
        yaml_cfg.add_attr("DILATION_KERNEL", [4, 4])

    if not hasattr(yaml_cfg, "MEDIAN_BLUR_1_KERNEL"):
        yaml_cfg.add_attr("MEDIAN_BLUR_1_KERNEL", 13)

    if not hasattr(yaml_cfg, "MEDIAN_BLUR_2_KERNEL"):
        yaml_cfg.add_attr("MEDIAN_BLUR_2_KERNEL", 7)

    return yaml_cfg


def resize_cfg(config, size, new_size):
    resized_cfg = {}
    resized_cfg["COLUMNS"] = [col * new_size[1] / size[1] for col in config.COLUMNS] 
    resized_cfg["ROW_OFFSET"] = config.ROW_OFFSET * new_size[0] / size[0]
    resized_cfg["COL_OFFSET"] = config.COL_OFFSET * new_size[1] / size[1]
    resized_cfg["ROWS"] = [[row[0]*new_size[0] / size[0], row[1]*new_size[0] / size[0]]  for row in  config.ROWS]
    resized_cfg["COL_RANGE"] = [ [range[0]* new_size[1] / size[1], range[1]* new_size[1] / size[1]]  for range in config.COL_RANGE]

    new_cfg = YamlCFG(resized_cfg)
    return new_cfg  



def estimate_noise(image):
    # return estimate_sigma(image, multichannel=False, average_sigmas=True)
    # min = np.min(image)
    # max = np.max(image)

    # image = (image - min) / (max - min)
    
    norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    image = norm_image.astype(np.uint8)

    # print(image)
    H, W = image.shape

    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(image, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return estimate_sigma(image, multichannel=False, average_sigmas=True)

def snr(image):
    # min = np.min(image)
    # max = np.max(image)

    # image = (image - min) / (max - min)

    signal = np.mean(image)
    noise = np.std(image)
    snr = signal / noise
    # 10 * np.log(np.abs(signal) / noise)   
    return noise