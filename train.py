import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  # Use CPU instead of MPS to avoid bicubic interpolation issues

import data
from model import UNetSR2x
from utils import loss_checkerboard_consistency, loss_dsu

from tqdm import tqdm
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=['MNIST', 'FashionMNIST', 'CelebA'])
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def save_images(model, dl, filename: str):
    model.eval()
    with torch.no_grad():
        x_vis, _ = next(iter(dl))
        x_vis = x_vis[:8].to(device)
        y_vis = model(x_vis).clamp(-1, 1)           # (B,C,2H,2W)
        grid_lr  = tv.utils.make_grid((x_vis+1)/2, nrow=4)
        grid_sr  = tv.utils.make_grid((y_vis+1)/2,  nrow=4)
        tv.utils.save_image(grid_lr, f"{filename}_lr.png")
        tv.utils.save_image(grid_sr, f"{filename}_sr.png")

if __name__ == "__main__":
    args = args_parser()

    lr = 2e-4
    epochs = 10

    # Get data
    dl_train, dl_val, channels = data.get_data(args.dataset, args.batch_size)
    
    # Model, diffusion
    model = UNetSR2x(in_ch=channels, base=16, out_ch=channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mode = "checkerboard"  # or "dsu"

    for epoch in tqdm(range(epochs)):

        # 
        model.train()
        for it, (x, _) in tqdm(enumerate(dl_train), total=len(dl_train)):
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

        # Save example images
        save_images(model, dl_val, filename=f"val_{args.dataset}_epoch{epoch}")
        save_images(model, dl_train, filename=f"train_{args.dataset}_epoch{epoch}")