import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules import Bottleneck, C3k, C3k2, Conv
except ImportError:
    pass 

class DEU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Bottleneck_DEU(Bottleneck):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.cv2 = DEU(int(c2 * e))

class C3k_DEU(C3k):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DEU(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DEU(C3k2):
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DEU(self.c, self.c, 2, shortcut, g) if c3k 
            else Bottleneck_DEU(self.c, self.c, shortcut, g) for _ in range(n)
        )

class FGMI(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_fft = torch.fft.fft2(self.dwconv1(x), norm='backward')
        x2_fft = torch.fft.fft2(self.dwconv2(x), norm='backward')
        out = torch.fft.ifft2(x1_fft * x2_fft, dim=(-2, -1), norm='backward').real
        return out * self.alpha + x * self.beta

class HFA(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        ker, pad = 31, 15
        
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)

        self.dw_13 = nn.Conv2d(dim, dim, (1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, (ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.act = nn.ReLU()

        self.conv = nn.Conv2d(dim, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, 1)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fgm = FGMI(dim)
        self.fusion_weights = nn.Parameter(torch.ones(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)

        x_fft = self.fac_conv(self.fac_pool(out)) * torch.fft.fft2(out, norm='backward')
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward').real

        x_sca = self.fgm(self.conv(self.pool(x_fca)) * x_fca)

        branch_feats = [self.dw_13(out), self.dw_31(out), self.dw_33(out), self.dw_11(out), x_sca]
        weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1)
        fused_feats = sum(w * feat for w, feat in zip(weights, branch_feats))

        return self.out_conv(self.act(x + fused_feats))

class CSPHFA(nn.Module):
    def __init__(self, dim: int, e: float = 1.0):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = HFA(int(dim * e)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = int(self.cv1.conv.out_channels * self.e)
        ok_branch, identity = torch.split(self.cv1(x), [c_, self.cv1.conv.out_channels - c_], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

class SPDConv(nn.Module):
    def __init__(self, inc: int, ouc: int, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            x[..., ::2, ::2], 
            x[..., 1::2, ::2], 
            x[..., ::2, 1::2], 
            x[..., 1::2, 1::2]
        ], 1)
        return self.conv(x)