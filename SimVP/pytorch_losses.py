import torch
import torch.nn as nn
import numpy as np
def interpolate_values(input_array, pixel_coords):
    device = input_array.device
    height = input_array.shape[-2]
    width = input_array.shape[-1]
    samples = input_array.shape[0]
    points = pixel_coords.shape[-2]
    
    # Pre-allocate the output tensor
    interpolated_values = torch.zeros(samples, points, device=device)

    # Get the integer parts and fractional parts of the pixel coordinates
    y_int = pixel_coords[..., 1].long()
    x_int = pixel_coords[..., 0].long()
    y_frac = pixel_coords[..., 1] - y_int.float()
    x_frac = pixel_coords[..., 0] - x_int.float()

    # Compute the indices of the neighboring pixels
    y0 = torch.clamp(y_int, 0, height - 2)
    y1 = y0 + 1
    x0 = torch.clamp(x_int, 0, width - 2)
    x1 = x0 + 1

    for i in range(samples):
        # Gather the pixel values at the neighboring indices
        values_y0x0 = input_array[i, 0, y0[i], x0[i]]
        values_y1x0 = input_array[i, 0, y1[i], x0[i]]
        values_y0x1 = input_array[i, 0, y0[i], x1[i]]
        values_y1x1 = input_array[i, 0, y1[i], x1[i]]

        # Perform bilinear interpolation
        interpolated_values[i] = (
            values_y0x0 * (1 - y_frac[i]) * (1 - x_frac[i]) +
            values_y1x0 * y_frac[i] * (1 - x_frac[i]) +
            values_y0x1 * (1 - y_frac[i]) * x_frac[i] +
            values_y1x1 * y_frac[i] * x_frac[i]
        )

    return interpolated_values

def torch_tracked_mse_interp(y_pred, y_true):
    eps = 1e-7
    device = y_pred.device

    losses = torch.zeros(y_pred.shape[1], device=device)
    
    for i in range(y_pred.shape[1]):
        data = y_pred[:, i]
        
        target_coords = y_true[:, i, :, :-1]  # Image pixel coordinates of withheld altimeter track
        
        # Perform linear interpolation using the interpolate_values function
        y_pred_loss = interpolate_values(data, target_coords)
        y_true_loss = y_true[:, i, :, -1]
        y_pred_loss *= (y_true_loss != 0).float()
        
        N_nz = torch.sum(y_true_loss != 0)
        N = N_nz + torch.sum(y_true_loss == 0)
        
        loss_loop = (N / (N_nz + eps)) * nn.MSELoss()(y_true_loss, y_pred_loss)
        
        losses[i] = loss_loop
    
    return torch.mean(losses)

def torch_tracked_mse_interp_weighted(y_pred, y_true,factor=3):
    #This is the same as torch_tracked_mse_interp but it weights the first few days more heavily
    eps = 1e-7
    device = y_pred.device

    losses = torch.zeros(y_pred.shape[1], device=device)
    weights=1-(((1-1/factor)*(1/y_pred.shape[1]))*np.arange(0,y_pred.shape[1]))#this makes the weight 1 on day 1 and 1/factor on day 30
    for i in range(y_pred.shape[1]):
        data = y_pred[:, i]
        
        target_coords = y_true[:, i, :, :-1]  # Image pixel coordinates of withheld altimeter track
        
        # Perform linear interpolation using the interpolate_values function
        y_pred_loss = interpolate_values(data, target_coords)
        y_true_loss = y_true[:, i, :, -1]
        y_pred_loss *= (y_true_loss != 0).float()
        
        N_nz = torch.sum(y_true_loss != 0)
        N = N_nz + torch.sum(y_true_loss == 0)
        
        loss_loop = (N / (N_nz + eps)) * nn.MSELoss()(y_true_loss, y_pred_loss)
        
        losses[i] = loss_loop*weights[i]
    return torch.mean(losses)

