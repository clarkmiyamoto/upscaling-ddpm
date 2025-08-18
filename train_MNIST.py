import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  # Use CPU instead of MPS to avoid bicubic interpolation issues

import data
from model import UNetSR2x
from utils import loss_checkerboard_consistency, loss_dsu

if __name__ == "__main__":
    channels = 1
    batch_size = 64
    lr = 2e-4
    epochs = 10

    # Get data
    dl = data.get_MNIST(batch_size)
    
    # Model, diffusion
    model = UNetSR2x(in_ch=1, base=16, out_ch=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mode = "checkerboard"  # or "dsu"

    for epoch in tqdm(range(epochs)):
        model.train()
        for it, (x, _) in tqdm(enumerate(dl), total=len(dl)):
            x = x.to(device)  # (B,C,H,W), arbitrary H,W if your loader provides it

            if mode == "checkerboard":
                y2x = model(x)
                # random parity each step
                loss = loss_checkerboard_consistency(y2x, x, parity=None)
            else:
                loss = loss_dsu(model, x, scale_half_mode="ceil")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (it+1) % 100 == 0:
                print(f"epoch {epoch} iter {it+1}: {loss.item():.4f}")

        # quick qualitative dump
        model.eval()
        with torch.no_grad():
            x_vis, _ = next(iter(dl))
            x_vis = x_vis[:8].to(device)
            y_vis = model(x_vis).clamp(-1, 1)           # (B,C,2H,2W)
            grid_lr  = tv.utils.make_grid((x_vis+1)/2, nrow=4)
            grid_sr  = tv.utils.make_grid((y_vis+1)/2,  nrow=4)
            tv.utils.save_image(grid_lr, f"lr_epoch{epoch}.png")
            tv.utils.save_image(grid_sr, f"sr_epoch{epoch}.png")