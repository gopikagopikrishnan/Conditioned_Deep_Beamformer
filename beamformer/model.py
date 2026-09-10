"""Intent-conditioned beamformer: anisotropic space-to-depth U-Net that predicts per-element receive apodization, with decoder-side FiLM conditioning on the intent vector q."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

# ---- locked baseline knobs (ablation wrappers override) --------------------
FH, FW  = 4, 2      # space-to-depth: pool axial 4x, lateral 2x
KH, KW  = 3, 7      # conv kernel (axial x lateral)
BASE_C  = 64


def _s2d(x, fh, fw):
    """Space-to-depth: (B,C,H,W) -> (B, C*fh*fw, H/fh, W/fw). Lossless downsample."""
    return rearrange(x, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=fh, w2=fw)


def _pad(kh, kw):
    return (kh // 2, kw // 2)


class AntiRectifier(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kh, kw, fh, fw, dropout=0.0):
        super().__init__()
        self.fh, self.fw = fh, fw
        self.conv1 = nn.Conv2d(in_ch,  out_ch // 2, (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch // 2); self.ar = AntiRectifier()
        self.conv2 = nn.Conv2d(out_ch, out_ch,     (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.ar(self.bn1(self.conv1(x)))
        skip = F.relu(self.drop(self.bn2(self.conv2(x))))
        down = _s2d(skip, self.fh, self.fw)
        return skip, down


class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kh, kw, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch // 2, (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch // 2); self.ar1 = AntiRectifier()
        self.conv2 = nn.Conv2d(out_ch, out_ch // 2, (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch // 2); self.ar2 = AntiRectifier()
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.ar1(self.bn1(self.conv1(x)))
        x = self.ar2(self.bn2(self.drop(self.conv2(x))))
        return x


class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kh, kw, fh, fw, dropout=0.0):
        super().__init__()
        up_ch = in_ch // 2; merged = up_ch + skip_ch
        self.up    = nn.ConvTranspose2d(in_ch, up_ch, kernel_size=(fh, fw), stride=(fh, fw))
        self.conv1 = nn.Conv2d(merged, out_ch // 2, (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch // 2); self.ar = AntiRectifier()
        self.conv2 = nn.Conv2d(out_ch, out_ch,     (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, skip, gamma=None, beta=None):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.ar(self.bn1(self.conv1(x)))
        x = F.relu(self.drop(self.bn2(self.conv2(x))))
        if gamma is not None and beta is not None:
            x = gamma * x + beta
        return x


class IntentConditioner(nn.Module):
    def __init__(self, ch_list, q_dim=2):
        super().__init__()
        self._CH = ch_list
        self.gamma_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(q_dim, ch), nn.GELU(), nn.Linear(ch, ch)) for ch in ch_list])
        self.beta_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(q_dim, ch), nn.GELU(), nn.Linear(ch, ch)) for ch in ch_list])
        for mlp in (*self.gamma_mlps, *self.beta_mlps):
            nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)

    def forward(self, q):
        e = q.view(-1, 2).float()
        gammas = [(mlp(e) + 1.0).unsqueeze(-1).unsqueeze(-1) for mlp in self.gamma_mlps]
        betas  = [mlp(e).unsqueeze(-1).unsqueeze(-1) for mlp in self.beta_mlps]
        return gammas, betas


class BeamformingHead(nn.Module):
    def __init__(self, in_channels, n_elements, kh, kw):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_elements * 2, (kh, kw), padding=_pad(kh, kw), bias=False)
        self.bn1   = nn.BatchNorm2d(n_elements * 2)
        self.conv2 = nn.Conv2d(n_elements * 2, n_elements, 1)   # 1x1 UNCHANGED
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.softmax(self.conv2(x), dim=1)


class SmallUNetBeamformerS2DFull(nn.Module):
    def __init__(self, n_elements=128, dropout=0.2,
                 kh=None, kw=None, base_c=None, fh=None, fw=None):
        super().__init__()
        kh = KH if kh is None else kh
        kw = KW if kw is None else kw
        fh = FH if fh is None else fh
        fw = FW if fw is None else fw
        c  = BASE_C if base_c is None else base_c
        self.n_elements = n_elements
        self.kh, self.kw, self.fh, self.fw, self.base_c = kh, kw, fh, fw, c
        f = fh * fw                                   # s2d channel expansion factor

        self.enc1 = EncBlock(n_elements, c,   kh, kw, fh, fw)
        self.enc2 = EncBlock(f * c,      2*c, kh, kw, fh, fw)
        self.enc3 = EncBlock(f * 2*c,    4*c, kh, kw, fh, fw)
        self.bottleneck = BottleneckBlock(f * 4*c, 8 * c, kh, kw, dropout=dropout)
        self.dec3 = DecBlock(8 * c, 4*c, 4*c, kh, kw, fh, fw, dropout=dropout)
        self.dec2 = DecBlock(4 * c, 2*c, 2*c, kh, kw, fh, fw, dropout=dropout)
        self.dec1 = DecBlock(2 * c, c,   c,   kh, kw, fh, fw, dropout=dropout)
        self.conditioner = IntentConditioner([4 * c, 2 * c, c], q_dim=2)
        self.head = BeamformingHead(c + 1, n_elements, kh, kw)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (BeamformingHead, IntentConditioner)):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x, q=None, q_mask=None):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x_mean = x.mean(dim=1, keepdim=True)
        skip1, d = self.enc1(x)
        skip2, d = self.enc2(d)
        skip3, d = self.enc3(d)
        d = self.bottleneck(d)
        q_eff = q
        if q is not None and q_mask is not None:
            q_eff = q * q_mask + torch.full_like(q, 0.5) * (1.0 - q_mask)
        if q_eff is not None:
            gammas, betas = self.conditioner(q_eff)
        else:
            gammas = betas = [None] * 3
        d = self.dec3(d, skip3, gammas[0], betas[0])
        d = self.dec2(d, skip2, gammas[1], betas[1])
        d = self.dec1(d, skip1, gammas[2], betas[2])
        return self.head(torch.cat([d, x_mean], dim=1))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Phase-0 verification: forward on a (1,128,128,128) TOFC tensor;
    # softmax weights must sum to 1 over the 128-element dim, no NaNs.
    for (fh, fw, kh, kw) in [(4, 2, 3, 7), (4, 1, 3, 7), (2, 2, 3, 3), (4, 2, 7, 3)]:
        m = SmallUNetBeamformerS2DFull(128, 0.2, kh=kh, kw=kw, fh=fh, fw=fw).eval()
        x = torch.randn(1, 128, 128, 128)
        q = torch.tensor([[1.0, 0.0]])
        with torch.no_grad():
            w = m(x, q=q)
        s = w.sum(dim=1)
        print(f"S2D({fh},{fw}) k{kh}x{kw}: {m.num_params:,} params | "
              f"out {tuple(w.shape)} sum~1: {torch.allclose(s, torch.ones_like(s), atol=1e-4)} "
              f"nan: {torch.isnan(w).any().item()}")
