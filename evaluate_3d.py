# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os 
import time
import argparse 

import torch

from torch.utils.data import DataLoader

from data.dataset_3d import Dataset3D
from data.plot import draw_grid 

from GAN.model_3d import Generator3d, Generator3d_2, ResidualGenerator3d_2
from GAN.train_3d import test
from GAN.loss import get_loss 

from config import combine_cfgs
from utils import init_torch_seeds, load_checkpoint
from experiments.tree_generator import TreeGenerator
from metrics.metrics import Metrics, tabulate_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=True)
    parser.add_argument('--model', type=str, help="Path to the trained model", required=False)
    parser.add_argument('--dataset_path', type=str, help="Path to Dataset", required=False)
    parser.add_argument('--split_dataset', type=bool, help="Specifies whether to split the dataset or not", required=False, default=False)
    parser.add_argument('--batch_size', type=int, help="Batch Size", required=False, default=1)
    parser.add_argument('--output_dir', type=str, help="Output directory", required=False, default="logs")

    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    model_path = args.model 
    split_dataset = args.split_dataset 
    output_dir = args.output_dir

    config = combine_cfgs(args.config)

    seed = config.SEED 
    batch_size = config.DATASET.BATCH_SIZE
    num_measurements = config.DATASET.NUM_MEASUREMENTS 
    normalize = config.DATASET.NORMALIZE 
    shuffle = config.DATASET.SHUFFLE
    standardize = config.DATASET.STANDARDIZE 
    smooth = config.DATASET.SMOOTH
    noise = config.DATASET.NOISE 
    noise_stdv = config.DATASET.NOISE_STDV
    global_scaling = config.DATASET.GLOBAL_SCALING 
    drop_zeros = config.DATASET.DROP_ZEROS 
    train_min = config.DATASET.TRAIN_MIN
    train_max = config.DATASET.TRAIN_MAX
    pos_value = config.DATASET.POS_VALUE
    neg_value = config.DATASET.NEG_VALUE
    batch_size = config.DATASET.BATCH_SIZE
    cap_meas = config.DATASET.CAP_MEAS
    train_split, val_split, test_split = config.DATASET.TRAIN_VAL_TEST_SPLIT    
    loss = config.SOLVER.LOSS 
    model_type = config.MODEL.TYPE
    head_activation = config.MODEL.HEAD_ACTIVATION
    hidden_activation = config.MODEL.HIDDEN_ACTIVATION 
    
    init_torch_seeds(seed)

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = config.DATASET.PATH

    if model_path is None: 
        save_path = os.path.join('experiments', exp_name)
        model_path = os.path.join(save_path, 'best_model.pth')
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_tree = TreeGenerator(root_dir=output_dir)
    output_tree.generate()

    # Prepare model and load parameters
    print(model_type)
    if model_type == 'Vanilla-Decoder':
        model = Generator3d_2(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation, num_channels=1)
    else:
        model = ResidualGenerator3d_2(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)

    model.load_state_dict(torch.load(model_path)["state_dict"])
    # model.load_state_dict(torch.load(model_path))
    # model, _, _ = load_checkpoint(model, gen_opt, output_tree.best_model_path)
    model = model.to(device)

    # Read dataset 
    print(shuffle)
    dataset = Dataset3D(dataset_path, shuffle=shuffle, normalize=normalize, standardize=standardize, global_scaling=global_scaling, drop_zeros=drop_zeros, smooth=smooth, pos_value=pos_value, neg_value=neg_value, 
                      train_min=train_min, train_max=train_max, cap_meas=cap_meas, device=device)

    if train_split != 0:
        split_dataset = True 
        
    if split_dataset:
        train_length = int(len(dataset)*train_split)
        val_length = int((len(dataset)*val_split))
        test_length = int((len(dataset) - train_length - val_length))
        _, _, test_dataset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length], generator=torch.Generator().manual_seed(seed))
    else: 
        test_dataset = dataset 

    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True) 


    gen_opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)


    # Prepare Loss 
    loss_fn = get_loss(loss)
    metrics = Metrics(device=device, metrics=['MSE', 'MAE', 'SSIM', 'PSNR', 'CC', 'IoU', 'MPA'])

    st = time.time()
    loss, metrics = test(model, loss_fn, test_loader, config, output_tree, device, metrics, save=True)
    run_time = time.time() - st / len(test_loader)
    print(run_time)

    # metrics = metrics.forward(predictions, ground_truth)
    print(metrics)

    save_path = os.path.join(output_dir, "stats.json")
    stats, table = tabulate_runs([metrics], run_time, save_path, use_gpu=False)
    print(table.draw())


if __name__ == "__main__":
    main()