
import os
import torch 
import shutil
import random
import numpy as np 

import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from matplotlib.patches import Circle


def plot_solution(u, save_filename, mesh_height):
    plt.imshow(u, cmap='viridis')
    plt.ylim([0, mesh_height])
    plt.colorbar()
    plt.savefig(save_filename)
    plt.close()
    plt.cla()
    # plt.show()

def plot_data_points(u, training_data, save_filename, mesh_height):
    fig, ax = plt.subplots(1)
    ax.imshow(u, cmap='viridis')

    for xx, yy in training_data:
        circ = Circle((xx, yy), 1, color='red')
        ax.add_patch(circ)
    ax.set_ylim([0, mesh_height])
    plt.savefig(save_filename)
    plt.close()
    plt.cla()
    # plt.show()


def normalize_input(inputs, mesh_width, mesh_height):
    normalized = inputs.clone().detach() 
    normalized[:, 0] = inputs[:, 0] / mesh_width  
    normalized[:, 1] = inputs[:, 1] / mesh_height
    return normalized 


def init_torch_seeds(seed):
    r""" Sets the seed for generating random numbers. Returns a
    Args:
        seed (int): The desired seed.
    """

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckp(state, is_best, checkpoint_dir, best_model_path):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, f_path)
    if is_best:
        shutil.copyfile(f_path, best_model_path)


def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch