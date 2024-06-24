import numpy as np
import torch

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
