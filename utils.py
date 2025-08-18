import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM
from typing import Tuple
import random
import math

def resize_bicubic(x, size):
    """x: (B,C,H,W). Returns anti-aliased bicubic resize to `size=(H,W)`."""
    return TF.resize(x, size, interpolation=IM.BICUBIC, antialias=True)

def resize_nearest(x, size):
    return TF.resize(x, size, interpolation=IM.NEAREST)

def match_spatial(x, ref):
    """Resize x to ref's (H,W) with bicubic AA."""
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W): return x
    return resize_bicubic(x, (H, W))

def make_checkerboard_mask(H, W, parity: Tuple[int, int], device):
    ii = torch.arange(H, device=device)[:, None]
    jj = torch.arange(W, device=device)[None, :]
    m = ((ii % 2 == parity[0]) & (jj % 2 == parity[1])).float()  # (H,W)
    return m[None, None]  # (1,1,H,W)

def loss_checkerboard_consistency(y_2x, x, parity=None):
    """
    y_2x: (B,C,2H,2W) predicted HR
    x:    (B,C,H,W)    LR input (normalized to [-1,1])
    Loss = MSE on a known checkerboard (nearest-upsampled LR) + downsample-consistency + TV
    """
    B, C, H, W = x.shape
    H2, W2 = y_2x.shape[-2:]

    # make targets at 2× resolution
    x_near_2x = resize_nearest(x, (H2, W2))
    if parity is None:
        parity = (random.randint(0,1), random.randint(0,1))
    m = make_checkerboard_mask(H2, W2, parity, device=x.device).expand(B, 1, H2, W2)

    # fidelity on known parity
    fidelity = F.mse_loss(y_2x * m, x_near_2x * m)

    # downsample-consistency (anti-aliased bicubic)
    x_back = resize_bicubic(y_2x, (H, W))
    recon = F.mse_loss(x_back, x)

    # light TV regularizer
    tv = (y_2x[:, :, :, 1:] - y_2x[:, :, :, :-1]).abs().mean() + \
         (y_2x[:, :, 1:, :] - y_2x[:, :, :-1, :]).abs().mean()

    return fidelity + 0.5 * recon + 1e-4 * tv

def loss_dsu(model, x, scale_half_mode="ceil"):
    """
    DSU: Downscale x by ~2 to x_half, feed x_half into model (which doubles),
         compare to original x (size-matched). Works for arbitrary sizes.
    """
    B, C, H, W = x.shape
    if scale_half_mode == "ceil":
        Hh, Wh = (math.ceil(H/2), math.ceil(W/2))
    elif scale_half_mode == "floor":
        Hh, Wh = (H//2, W//2)
    else:
        r = 0.5
        Hh, Wh = (max(1, int(round(H*r))), max(1, int(round(W*r))))

    x_half = resize_bicubic(x, (Hh, Wh))
    y_pred = model(x_half)                  # ~ (2Hh, 2Wh)
    y_pred = match_spatial(y_pred, x)       # ensure (H,W)

    # primary reconstruction loss + consistency back to half
    recon_full = F.mse_loss(y_pred, x)
    recon_half = F.mse_loss(resize_bicubic(y_pred, (Hh, Wh)), x_half)

    tv = (y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]).abs().mean() + \
         (y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]).abs().mean()

    return recon_full + 0.25 * recon_half + 1e-4 * tv