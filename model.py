import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from utils import match_spatial



class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.op(x)

class ResBlock(nn.Module):
    def __init__(self, c, expansion=2/3):
        super().__init__()
        mid = max(8, int(c * expansion))
        self.block = nn.Sequential(
            nn.Conv2d(c, mid, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 3, 1, 1), nn.BatchNorm2d(c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x): return self.act(x + self.block(x))

class Down(nn.Module):
    """Stride-2 conv; output ~ ceil(H/2)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.op = nn.Sequential(
            ConvBNAct(c_in, c_out, 3, 2, 1),
            ResBlock(c_out),
        )
    def forward(self, x): return self.op(x)

class UpShuffle(nn.Module):
    """Up by 2 with PixelShuffle to avoid checkerboard artifacts."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pre = nn.Conv2d(c_in, c_out * 4, 3, 1, 1)
        self.post = nn.Sequential(nn.SiLU(inplace=True), ResBlock(c_out))

    def forward(self, x):
        x = self.pre(x)
        x = F.pixel_shuffle(x, 2)
        return self.post(x)

class UNetSR2x(jit.ScriptModule):
    """
    Input:  (B,C,H,W)
    Output: (B,C,2H,2W)
    Works for arbitrary H,W (odd/even).
    """
    def __init__(self, in_ch=3, base=96, out_ch=3):
        super().__init__()
        # encoder
        self.stem = nn.Sequential(ConvBNAct(in_ch, base), ResBlock(base))
        self.down1 = Down(base, base*2)   # H -> ~ceil(H/2)
        self.down2 = Down(base*2, base*4) # -> ~ceil(H/4)

        # bottleneck
        self.mid = nn.Sequential(ResBlock(base*4), ResBlock(base*4))

        # decoder
        self.up1 = UpShuffle(base*4, base*2)  # ~x2
        self.up2 = UpShuffle(base*2, base)    # ~x2 back to ~input size

        # final 2× stage (to exactly 2H,2W later)
        self.to2x = UpShuffle(base, base)
        self.head = nn.Sequential(nn.Conv2d(base, out_ch, 3, 1, 1), nn.Tanh())

    @jit.script_method
    def forward(self, x):
        B,C,H,W = x.shape

        # encoder
        s0 = self.stem(x)          # (B,b,H,W)
        s1 = self.down1(s0)        # (B,2b,~H/2,~W/2)
        s2 = self.down2(s1)        # (B,4b,~H/4,~W/4)

        # bottleneck
        h = self.mid(s2)

        # decoder with shape-matching to avoid off-by-one issues
        h = self.up1(h)            # -> ~H/2
        h = match_spatial(h, s1); h = h + s1

        h = self.up2(h)            # -> ~H
        h = match_spatial(h, s0); h = h + s0

        # final 2×
        h = self.to2x(h)           # -> ~2H
        # ensure *exactly* (2H, 2W)
        h = match_spatial(h, torch.empty(B,1,2*H,2*W, device=h.device))
        y = self.head(h)           # (B,C,2H,2W), scaled to [-1,1]
        return y
