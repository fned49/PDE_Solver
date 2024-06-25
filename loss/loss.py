import torch

def unsupervised_loss(u_pred, f, h, interior_mask, boundary_mask):
    laplacian_filter = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(u_pred.device) # (1, 1, H, W) 형태로 변환 / 마이너스 라플라시안 필터
    laplacian_u = torch.nn.functional.conv2d(u_pred, laplacian_filter, padding=1)

    interior_loss = torch.sum(((f * h**2 - laplacian_u) ** 2) * interior_mask)
    boundary_loss = torch.sum((u_pred * boundary_mask) ** 2)

    return interior_loss + boundary_loss

def h_loss(u_pred, f, h, interior_mask, boundary_mask):
    laplacian_filter = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(u_pred.device) # (1, 1, H, W) 형태로 변환 / 마이너스 라플라시안 필터
    laplacian_u = torch.nn.functional.conv2d(u_pred, laplacian_filter, padding=1)

    interior_loss = torch.sum(((f * h**2 - laplacian_u) ** 2) * interior_mask)
    boundary_loss = torch.sum((u_pred * boundary_mask) ** 2)

    return h**4*interior_loss + boundary_loss/h