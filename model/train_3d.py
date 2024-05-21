import os
import sys 
import time 
import pickle
import tifffile
import torch 
torch.cuda.empty_cache()

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from torch.autograd import Variable

from data.plot import draw_grid, draw_confocal_grid, draw_line, plot_3d_volume


def train_one_epoch(generator, optimizer, loss_fn, train_loader, val_loader, epoch, device, noise=False, noise_stdv=0.02):
    train_losses = 0
    val_losses = 0

    for x, y in train_loader: 
        perm  = x["perm_xy"]
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        optimizer.zero_grad()

        generator.train(True)

        # perturb input with gaussian noise
        if noise:
            vb =  vb + Variable(torch.randn(vb.shape, device=device) * noise_stdv)

        input_g = vb.view(vb.shape[0], vb.shape[1], vb.shape[2], 1, 1)
        predicted_perm = generator(input_g.float())
        predicted_perm = predicted_perm.to(device)
        
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                        for p in generator.parameters())

        perm = perm.view(perm.shape[0], 1, perm.shape[1], perm.shape[2], perm.shape[3])

        loss = loss_fn(predicted_perm, perm)

        loss.backward()
        optimizer.step()

        train_losses += loss.item()

    generator.train(False)

    for x, y in val_loader:
        perm  = x["perm_xy"]
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        # perturb input with gaussian noise
        if noise:
            vb = vb + Variable(torch.randn(vb.shape, device=device) * noise_stdv)
            
        # input_g = vb.view(vb.shape[0], vb.shape[1], 1, 1)
        input_g = vb.view(vb.shape[0], vb.shape[1], vb.shape[2], 1, 1)

        predicted_perm = generator(input_g.float())
        perm = perm.view(perm.shape[0], 1, perm.shape[1], perm.shape[2], perm.shape[3])

        loss = loss_fn(predicted_perm, perm) 
        val_losses += loss.item() 
   
    train_avg_loss = train_losses / len(train_loader)
    val_avg_loss = val_losses / len(val_loader)

    print('Epoch: %0.2f | Training Loss: %.6f | Validation Loss: %0.6f'  % (epoch, train_avg_loss, val_avg_loss), flush=True)
    # print(loss_fn.awl.params, flush=True)

    return train_avg_loss, val_avg_loss 


def train_adv(generator, discrimantor, optimizerG, optimizerD, training_loader, val_loader, epoch, device):
    real_label = 1
    fake_label = 0
    nz = 100
    G_losses = []
    D_losses = []

    criterion = torch.nn.BCELoss()
    for x, y in training_loader: 
        perm  = x["perm_xy"]
        vb = y["v_b"] 

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discrimantor.zero_grad()
        # Format batch
        real_perm = perm.to(device)
        b_size = real_perm.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = discrimantor(real_perm).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        
        # Classify all fake batch with D
        output = discrimantor(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discrimantor(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    return G_losses, D_losses

    
def test(generator, loss_fn, test_loader, config, output_tree, device, metrics, save=True):
    # predictions = torch.tensor([])
    # predictions = predictions.to(device)

    # ground_truth = torch.tensor([])
    # ground_truth = ground_truth.to(device)

    generator.train(False)

    metrics_list = dict()

    for i, (x, y) in enumerate(tqdm(test_loader)):
        perm  = x["perm_xy"].float()
        vb = y["v_b"].float()

        perm = perm.to(device)
        vb = vb.to(device)

        input_g = vb.view(vb.shape[0], vb.shape[1], vb.shape[2], 1, 1)

        st = time.time()
        predicted_perm = generator(input_g)
        predicted_perm = predicted_perm.to(device)
        end = time.time() - st 

        # smooth predictions
        predicted_perm = predicted_perm.reshape(perm.shape)
        pred_perm_smoothed, ground_truth_smoothed = smooth_predictions(predicted_perm, perm, config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)

        metrics_values = metrics.forward(pred_perm_smoothed, ground_truth_smoothed)

        for k, val in metrics_values.items(): 
            if k in metrics_list:
                metrics_list[k] += val.item()
            else:
                metrics_list[k] = val.item()

        if save: 
            for j, pred in enumerate(pred_perm_smoothed): 
                pred_perm = pred
                truth_perm = ground_truth_smoothed[j]

                # plot max projection 
                pred_max, _ = torch.max(pred_perm, dim=0)
                truth_max, _ = torch.max(truth_perm, dim=0)

                draw_grid(pred_max.cpu().detach().numpy(), "ECT Prediction", "Row (200\u03bcm)", "Depth (100\u03bcm)",  os.path.join(output_tree.pred_path, f"max_proj_pred_{i}_{j}.png"), font_size=24, figsize=(6, 6), cmap='viridis',  ticks=False, scale_bar=True, colorbar=False)
                draw_confocal_grid(truth_max.cpu().detach().numpy(), "Ground Truth", "Row (y)", "Depth (z)",  os.path.join(output_tree.true_path, f"max_proj_truth_{i}_{j}.png"), font_size=24, figsize=(6, 6), cmap='Reds', ticks=False, scale_bar=True, colorbar=False)
                
                # save 3-D predictions and ground truth 
                save_path =  os.path.join(output_tree.pred_path, f"pred_{i}_{j}.tif")
                pred_perm_scaled = scale_volume_x(pred_perm.cpu().detach().numpy())
                tifffile.imwrite(save_path, pred_perm_scaled, compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)
               
                # with open(save_path, 'wb') as f:
                #     # np.save(f, self.v_b)
                #     pickle.dump(pred_perm.cpu().detach().numpy(), f)

                save_path =  os.path.join(output_tree.pred_path, f"truth_{i}_{j}.tif")
                truth_perm_scaled = scale_volume_x(truth_perm.cpu().detach().numpy())
                tifffile.imwrite(save_path, truth_perm_scaled, compression='zlib', metadata={'axes': 'CYX', 'mode': 'composite'}, imagej=True)

                # with open(save_path, 'wb') as f:
                #     # np.save(f, self.v_b)
                #     pickle.dump(truth_perm.cpu().detach().numpy(), f)
    
    for k, values in metrics_list.items():
        metrics_list[k] = values / len(test_loader)


    return metrics_list
# , predictions, ground_truth

def scale_volume_x(volume, dsize=(50, 100, 200)): 
    x_old = np.linspace(0, (volume.shape[1]-1)*1, volume.shape[1])
    y_old = np.linspace(0, (volume.shape[2]-1)*1, volume.shape[2])
    z_old = np.arange(0, (volume.shape[0]))*1
    slice_thickness_new = 1

    my_interpolating_object = RegularGridInterpolator((z_old, x_old, y_old), volume, method="linear", bounds_error=False, fill_value=0)
    
    x_new = np.round(volume.shape[1]*1/1).astype('int')
    y_new = np.round(volume.shape[2]*1/1).astype('int')
    z_new = np.arange(z_old[0], 50, slice_thickness_new)

    pts = np.indices((len(z_new), x_new, y_new)).transpose((1, 2, 3, 0))
    pts = pts.reshape(1, len(z_new)*x_new*y_new, 1, 3).reshape(len(z_new)*x_new*y_new, 3)
    pts = np.array(pts, dtype=float)
    pts[:, 1:3] = pts[:, 1:3]*1
    pts[:, 0] = pts[:, 0]*slice_thickness_new +z_new[0]
        

    interpolated_data = my_interpolating_object(pts)
    interpolated_data = interpolated_data.reshape(len(z_new), x_new, y_new)

    zoom = [10, 1, 1]
    interpolated_data = ndimage.interpolation.zoom(volume, zoom=zoom) 

    x, z, y = interpolated_data.shape
    interpolated_data_reshaped = np.zeros((z, y, x), dtype=volume.dtype)
    
    for i in range(z): 
        interpolated_data_reshaped[i, :, :] = interpolated_data[:, i, :].reshape((y, x))
    
    # print(interpolated_data_reshaped.shape)
    return np.array(interpolated_data, dtype=volume.dtype)


def smooth_predictions(predicted_perm, ground_truth, activation, pos_value, neg_value):
    pred_perm = torch.clone(predicted_perm)
    truth_perm = torch.clone(ground_truth)

    if activation == 'Tanh':
        pred_perm[predicted_perm < 0] = 1 # pos_value
        pred_perm[predicted_perm >= 0] = 0 # neg_value
        truth_perm[ground_truth < 0] = 1 
        truth_perm[ground_truth >= 0] = 0 
        
        
    if activation == 'Sigmoid':
        pred_perm[predicted_perm >= 0.45] = pos_value
        pred_perm[predicted_perm < 0.45] = neg_value


    return pred_perm, truth_perm