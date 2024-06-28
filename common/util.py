import numpy as np
import torch
import torch.nn as nn

def Dirichlet_bc_array(f_matrix, g=0):
    f = f_matrix.copy()
    f[:, 0] = g  # 왼쪽 경계
    f[:, -1] = g  # 오른쪽 경계
    f[0, :] = g  # 아래쪽 경계
    f[-1, :] = g  # 위쪽 경계
    return f

def Dirichlet_bc_tensor(u_pred, g=0):
    u = u_pred.clone()
    u[:, :, 0, :] = g  # 왼쪽 경계
    u[:, :, -1, :] = g  # 오른쪽 경계
    u[:, :, :, 0] = g  # 아래쪽 경계
    u[:, :, :, -1] = g  # 위쪽 경계
    return u

def create_boundary_mask(size):
    mask = torch.zeros(size, dtype=torch.float32)
    mask[:, 0] = 1  # 왼쪽 경계
    mask[:, -1] = 1  # 오른쪽 경계
    mask[0, :] = 1  # 아래쪽 경계
    mask[-1, :] = 1  # 위쪽 경계
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) 형태로 변환

def create_interior_mask(size):
    mask = torch.ones(size, dtype=torch.float32)
    mask[:, 0] = 0  # 왼쪽 경계
    mask[:, -1] = 0  # 오른쪽 경계
    mask[0, :] = 0  # 아래쪽 경계
    mask[-1, :] = 0  # 위쪽 경계
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) 형태로 변환

def he_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
