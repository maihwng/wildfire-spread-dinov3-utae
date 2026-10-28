"""
================================================================================
DINOv3-UTAE: Spatio-Temporal Wildfire Spread Prediction Model
================================================================================
Architecture combining:
  1. Meta's DINOv3 ViT-S/16 foundation model as the spatial feature extractor.
  2. 17-channel wildfire multi-modal input processing (weather, topography, vegetation, fire history).
  3. L-TAE (Lightweight Temporal Attention Encoder) for multi-temporal sequence fusion.
  4. Temporal Skip Aggregation and U-TAE multi-scale convolutional decoder.
  5. Output: (B, 1, 128, 128) next-day wildfire spread probability map / logits.
================================================================================
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Dynamic Path Resolution: ensure dinov3 and utae-paps modules can be imported
# -----------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent          # wildfire-prediction
_WORKSPACE_ROOT = _PROJECT_ROOT.parent        # TH2

for _p in [
    str(_WORKSPACE_ROOT / "dinov3"),
    str(_WORKSPACE_ROOT / "utae-paps"),
    str(_WORKSPACE_ROOT),
    str(_PROJECT_ROOT),
]:
    if _p not in sys.path and Path(_p).exists():
        sys.path.insert(0, _p)

# Import DINOv3 components
try:
    from dinov3.models.vision_transformer import DinoVisionTransformer
except ImportError:
    try:
        from dinov3.hub.backbones import dinov3_vits16
    except ImportError:
        DinoVisionTransformer = None


# =============================================================================
# Helper: Checkpoint Discovery
# =============================================================================
def find_checkpoint(checkpoint_path: Optional[str] = None) -> Optional[str]:
    """
    Search for DINOv3 ViT-S/16 pretrained checkpoint (.pth) in common locations.
    """
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        return os.path.abspath(checkpoint_path)

    candidate_files = [
        checkpoint_path,
        "checkpoints/vits16.pth",
        "wildfire-prediction/checkpoints/vits16.pth",
        "../checkpoints/vits16.pth",
        "../../checkpoints/vits16.pth",
        "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "wildfire-prediction/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "../checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ]
    for c in candidate_files:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    # Search directory trees
    search_dirs = [
        Path.cwd() / "checkpoints",
        Path.cwd() / "wildfire-prediction" / "checkpoints",
        _PROJECT_ROOT / "checkpoints",
        _WORKSPACE_ROOT / "wildfire-prediction" / "checkpoints",
    ]
    for sdir in search_dirs:
        if sdir.is_dir():
            target = sdir / "vits16.pth"
            if target.is_file():
                return str(target.resolve())
            for f in sdir.glob("*.pth"):
                if "vits16" in f.name.lower() or "dinov3" in f.name.lower():
                    return str(f.resolve())
    return None


# =============================================================================
# Building Blocks: Convolutional Layers & Blocks (Compatible with U-TAE)
# =============================================================================
class ConvLayer(nn.Module):
    """
    Multi-layer 2D convolution with normalization (batch, group, instance) and ReLU.
    """
    def __init__(
        self,
        nkernels: List[int],
        norm: str = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=min(n_groups, num_feats),
            )
        else:
            nl = None

        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU(inplace=True))
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    """
    Shared convolutional block supporting both 4D (B, C, H, W) and 5D (B, T, C, H, W) inputs.
    """
    def __init__(
        self,
        nkernels: List[int],
        norm: str = "batch",
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super(ConvBlock, self).__init__()
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            out = self.conv(x.view(b * t, c, h, w))
            _, c_out, h_out, w_out = out.shape
            return out.view(b, t, c_out, h_out, w_out)
        return self.conv(x)


class DownConvBlock(nn.Module):
    """
    Strided downsampling block with residual skip connection.
    Supports both 4D and 5D inputs.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 4,
        s: int = 2,
        p: int = 1,
        norm: str = "batch",
        padding_mode: str = "reflect",
    ):
        super(DownConvBlock, self).__init__()
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_5d = x.dim() == 5
        if is_5d:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)

        out = self.down(x)
        out = self.conv1(out)
        out = out + self.conv2(out)

        if is_5d:
            _, c_out, h_out, w_out = out.shape
            out = out.view(b, t, c_out, h_out, w_out)
        return out


class UpConvBlock(nn.Module):
    """
    Transposed convolution upsampling block with skip connection fusion.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 4,
        s: int = 2,
        p: int = 1,
        norm: str = "batch",
        d_skip: Optional[int] = None,
        padding_mode: str = "reflect",
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(inplace=True),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(x)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


# =============================================================================
# Temporal Attention: L-TAE (Lightweight Temporal Attention Encoder)
# =============================================================================
class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding for temporal steps.
    Supports either relative day index [1..T] or day of year.
    """
    def __init__(self, d: int, T: int = 1000, repeat: Optional[int] = None, offset: int = 0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.register_buffer("denom", denom)

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        # batch_positions: (B*H*W, T)
        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)
        return sinusoid_table


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: (n_head, B*H*W, 1, d_k), k: (n_head, B*H*W, T, d_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # (n_head, B*H*W, 1, T)
        if pad_mask is not None:
            # pad_mask: (B*H*W, T)
            attn = attn.masked_fill(pad_mask[None, :, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)  # (n_head, B*H*W, 1, d_in/n_head)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Master Query Multi-Head Attention module from L-TAE.
    Uses a single learned master query vector Q per head.
    """
    def __init__(self, n_head: int, d_k: int, d_in: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=math.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=math.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B*H*W, T, d_in)
        sz_b, len_s, _ = x.shape
        v = x.view(sz_b, len_s, self.n_head, -1).permute(2, 0, 1, 3)  # (n_head, sz_b, T, d_v)
        k = self.fc1_k(x).view(sz_b, len_s, self.n_head, self.d_k).permute(2, 0, 1, 3)

        q = self.Q[:, None, None, :].expand(self.n_head, sz_b, 1, self.d_k)
        out, attn = self.attention(q, k, v, pad_mask=pad_mask)
        out = out.squeeze(2)      # (n_head, sz_b, d_v)
        attn = attn.squeeze(2)    # (n_head, sz_b, T)
        return out, attn


class LTAE2d(nn.Module):
    """
    Lightweight Temporal Attention Encoder (L-TAE) for multi-temporal feature maps.
    Aggregates (B, T, C, H, W) -> (B, C_out, H, W) and generates attention masks.
    """
    def __init__(
        self,
        in_channels: int = 384,
        n_head: int = 16,
        d_k: int = 4,
        mlp: Optional[List[int]] = None,
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1000,
        return_att: bool = True,
        positional_encoding: bool = True,
    ):
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.return_att = return_att
        mlp = mlp or [d_model, in_channels]
        self.mlp_dims = mlp

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers = []
        for i in range(len(mlp) - 1):
            layers.extend([
                nn.Linear(mlp[i], mlp[i + 1]),
                nn.BatchNorm1d(mlp[i + 1]),
                nn.ReLU(inplace=True),
            ])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C, H, W)
        sz_b, seq_len, d, h, w = x.shape

        if pad_mask is not None:
            pad_mask_expanded = (
                pad_mask.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(sz_b, seq_len, h, w)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(sz_b * h * w, seq_len)
            )
        else:
            pad_mask_expanded = None

        # Reshape to (B*H*W, T, C)
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        # Positional Encoding
        if self.positional_encoder is not None:
            if batch_positions is None:
                batch_positions = (
                    torch.arange(1, seq_len + 1, device=x.device)
                    .unsqueeze(0)
                    .expand(sz_b, -1)
                )
            bp = (
                batch_positions.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(sz_b, seq_len, h, w)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(sz_b * h * w, seq_len)
            )
            out = out + self.positional_encoder(bp)

        # Temporal Multi-Head Attention
        out, attn = self.attention_heads(out, pad_mask=pad_mask_expanded)

        # Output projection
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.dropout(self.mlp(out))
        if self.out_norm is not None:
            out = self.out_norm(out)
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (B, C_out, H, W)

        # Reshape attention map: (n_head, B, T, H, W)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3).contiguous()

        if self.return_att:
            return out, attn
        return out


# =============================================================================
# Temporal Aggregator for Skip Connections
# =============================================================================
class Temporal_Aggregator(nn.Module):
    """
    Fuses multi-scale skip connections across time using the L-TAE attention masks.
    """
    def __init__(self, mode: str = "att_group"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, C, H, W), attn_mask: (n_head, B, T, H_att, W_att)
        if self.mode == "att_group" and attn_mask is not None:
            n_heads, b, t, h_att, w_att = attn_mask.shape
            attn = attn_mask.view(n_heads * b, t, h_att, w_att)

            if x.shape[-2] != h_att or x.shape[-1] != w_att:
                attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)

            attn = attn.view(n_heads, b, t, *x.shape[-2:])

            if pad_mask is not None and pad_mask.any():
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

            # Split channels into n_heads groups
            c = x.shape[2]
            assert c % n_heads == 0, f"Skip channel count {c} must be divisible by n_heads {n_heads}"
            groups = torch.stack(x.chunk(n_heads, dim=2))  # (n_head, B, T, C/n_head, H, W)
            out = (attn[:, :, :, None, :, :] * groups).sum(dim=2)  # (n_head, B, C/n_head, H, W)
            out = torch.cat([g for g in out], dim=1)               # (B, C, H, W)
            return out

        elif self.mode == "att_mean" and attn_mask is not None:
            attn = attn_mask.mean(dim=0)  # (B, T, H_att, W_att)
            if x.shape[-2] != attn.shape[-2] or x.shape[-1] != attn.shape[-1]:
                attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if pad_mask is not None and pad_mask.any():
                attn = attn * (~pad_mask).float()[:, :, None, None]
            out = (x * attn[:, :, None, :, :]).sum(dim=1)
            return out

        else:
            if pad_mask is not None and pad_mask.any():
                mask = (~pad_mask).float()[:, :, None, None, None]
                out = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                return out
            return x.mean(dim=1)


# =============================================================================
# 17-Channel Multi-Modal Input Adapter
# =============================================================================
class WildfireInputAdapter(nn.Module):
    """
    Adapts 17 multi-modal wildfire features (elevation, slope, aspect, wind velocity,
    direction, temperature, humidity, precipitation, vegetation, NDVI, fire masks, etc.)
    into 3 pseudo-RGB channels for the pretrained DINOv3 ViT-S/16 backbone.
    """
    def __init__(self, in_channels: int = 17, out_channels: int = 3, mode: str = "conv"):
        super().__init__()
        self.mode = mode
        if mode == "conv":
            # 3x3 conv with GroupNorm and GELU for expressive non-linear channel fusion
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(num_groups=4, num_channels=32),
                nn.GELU(),
                nn.Conv2d(32, out_channels, kernel_size=1),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.GELU(),
            )
        elif mode == "linear":
            # Direct 1x1 learned linear channel mapping
            self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


# =============================================================================
# Main Model: DINOv3-UTAE Architecture
# =============================================================================
class DINOv3_UTAE(nn.Module):
    """
    DINOv3-UTAE: Spatio-Temporal Segmentation Model for Wildfire Spread Prediction.

    Combines:
      - Spatial Encoder: Meta's DINOv3 ViT-S/16 (pretrained on LVD-1689M).
      - Multi-scale stem and deconvolution pyramid for high-resolution skip connections.
      - 17-channel wildfire input adapter.
      - L-TAE (Lightweight Temporal Attention Encoder) at the 8x8 bottleneck.
      - Temporal attention skip connection aggregator.
      - U-TAE 4-stage convolutional decoder upsampling to (B, 1, 128, 128).
    """
    def __init__(
        self,
        in_channels: int = 17,
        n_classes: int = 1,
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        channel_strategy: str = "adapter",  # 'adapter', 'conv1x1', or 'expand'
        encoder_widths: List[int] = [64, 64, 128, 256, 384],
        decoder_widths: List[int] = [32, 64, 128, 256, 384],
        out_conv: List[int] = [32],
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        decoder_norm: str = "batch",
        pad_value: float = 0.0,
        padding_mode: str = "reflect",
    ):
        super(DINOv3_UTAE, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.pad_value = pad_value
        self.channel_strategy = channel_strategy
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.n_head = n_head
        self.d_model = d_model

        # ---------------------------------------------------------------------
        # 1. 17-Channel Input Adapter for DINOv3
        # ---------------------------------------------------------------------
        if channel_strategy in ("adapter", "conv"):
            self.input_adapter = WildfireInputAdapter(in_channels=in_channels, out_channels=3, mode="conv")
            dino_in_chans = 3
        elif channel_strategy == "conv1x1":
            self.input_adapter = WildfireInputAdapter(in_channels=in_channels, out_channels=3, mode="linear")
            dino_in_chans = 3
        elif channel_strategy == "expand":
            self.input_adapter = nn.Identity()
            dino_in_chans = in_channels
        else:
            raise ValueError(f"Unknown channel_strategy: {channel_strategy}")

        # ---------------------------------------------------------------------
        # 2. DINOv3 ViT-S/16 Spatial Backbone
        # ---------------------------------------------------------------------
        self.dino = self._build_dinov3_backbone(
            in_chans=dino_in_chans,
            checkpoint_path=checkpoint_path,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for p in self.dino.parameters():
                p.requires_grad = False

        # ---------------------------------------------------------------------
        # 3. High-Resolution Spatial CNN Stem for U-Net Skip Connections
        # ---------------------------------------------------------------------
        self.stem0 = ConvBlock(
            nkernels=[in_channels, encoder_widths[0], encoder_widths[0]],
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down1 = DownConvBlock(
            d_in=encoder_widths[0],
            d_out=encoder_widths[1],
            k=4, s=2, p=1,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down2 = DownConvBlock(
            d_in=encoder_widths[1],
            d_out=encoder_widths[2],
            k=4, s=2, p=1,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down3 = DownConvBlock(
            d_in=encoder_widths[2],
            d_out=encoder_widths[3],
            k=4, s=2, p=1,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down4 = DownConvBlock(
            d_in=encoder_widths[3],
            d_out=encoder_widths[4],
            k=4, s=2, p=1,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )

        # ---------------------------------------------------------------------
        # 4. Multi-Scale Deconvolution Projections for DINOv3 Intermediate Layers
        #    DINOv3 produces (B*T, 384, 8, 8) at blocks [2, 5, 8, 11].
        #    We project them to match the resolutions of skip levels:
        #    - Block 2 (depth 3)   -> 64x64   (stride 8 deconv)
        #    - Block 5 (depth 6)   -> 32x32   (stride 4 deconv)
        #    - Block 8 (depth 9)   -> 16x16   (stride 2 deconv)
        #    - Block 11 (depth 12) -> 8x8     (1x1 conv bottleneck)
        # ---------------------------------------------------------------------
        embed_dim = 384
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, encoder_widths[1], kernel_size=8, stride=8),
            nn.BatchNorm2d(encoder_widths[1]),
            nn.GELU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, encoder_widths[2], kernel_size=4, stride=4),
            nn.BatchNorm2d(encoder_widths[2]),
            nn.GELU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, encoder_widths[3], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_widths[3]),
            nn.GELU(),
        )
        self.proj_bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, encoder_widths[4], kernel_size=1),
            nn.BatchNorm2d(encoder_widths[4]),
            nn.GELU(),
        )

        # ---------------------------------------------------------------------
        # 5. Temporal Encoder: L-TAE (Lightweight Temporal Attention Encoder)
        # ---------------------------------------------------------------------
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[4],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[4]],
            return_att=True,
            d_k=d_k,
        )

        # ---------------------------------------------------------------------
        # 6. Temporal Aggregator for Skip Connections
        # ---------------------------------------------------------------------
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # ---------------------------------------------------------------------
        # 7. Spatial Decoder (4-Stage U-TAE Decoder)
        # ---------------------------------------------------------------------
        # Up 0: 8x8 -> 16x16 (fuse skip3)
        self.up0 = UpConvBlock(
            d_in=decoder_widths[4],
            d_out=decoder_widths[3],
            d_skip=encoder_widths[3],
            k=4, s=2, p=1,
            norm=decoder_norm,
            padding_mode=padding_mode,
        )
        # Up 1: 16x16 -> 32x32 (fuse skip2)
        self.up1 = UpConvBlock(
            d_in=decoder_widths[3],
            d_out=decoder_widths[2],
            d_skip=encoder_widths[2],
            k=4, s=2, p=1,
            norm=decoder_norm,
            padding_mode=padding_mode,
        )
        # Up 2: 32x32 -> 64x64 (fuse skip1)
        self.up2 = UpConvBlock(
            d_in=decoder_widths[2],
            d_out=decoder_widths[1],
            d_skip=encoder_widths[1],
            k=4, s=2, p=1,
            norm=decoder_norm,
            padding_mode=padding_mode,
        )
        # Up 3: 64x64 -> 128x128 (fuse skip0)
        self.up3 = UpConvBlock(
            d_in=decoder_widths[1],
            d_out=decoder_widths[0],
            d_skip=encoder_widths[0],
            k=4, s=2, p=1,
            norm=decoder_norm,
            padding_mode=padding_mode,
        )

        # ---------------------------------------------------------------------
        # 8. Output Convolutional Head: -> (B, n_classes, 128, 128)
        # ---------------------------------------------------------------------
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv + [n_classes],
            norm=decoder_norm,
            last_relu=False,
            padding_mode=padding_mode,
        )

    def _build_dinov3_backbone(
        self,
        in_chans: int = 3,
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
    ) -> nn.Module:
        """
        Instantiate DINOv3 ViT-S/16 architecture and load pretrained weights.
        """
        model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3 if self.channel_strategy != "expand" else in_chans,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2.0,
            pos_embed_rope_dtype="fp32",
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )

        resolved_ckpt = find_checkpoint(checkpoint_path) if pretrained else None
        if resolved_ckpt:
            print(f"[DINOv3-UTAE] Loading pretrained weights from: {resolved_ckpt}")
            state_dict = torch.load(resolved_ckpt, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            if self.channel_strategy == "expand" and in_chans != 3:
                # Inflate patch_embed.proj from 3 channels to in_chans
                orig_w = state_dict["patch_embed.proj.weight"]  # [384, 3, 16, 16]
                new_w = torch.zeros(384, in_chans, 16, 16, dtype=orig_w.dtype)
                new_w[:, :3, :, :] = orig_w
                # Replicate mean weight across remaining channels scaled to maintain variance
                new_w[:, 3:, :, :] = orig_w.mean(dim=1, keepdim=True).repeat(1, in_chans - 3, 1, 1) * (3.0 / in_chans)
                state_dict["patch_embed.proj.weight"] = new_w
                model.patch_embed.proj = nn.Conv2d(in_chans, 384, kernel_size=16, stride=16)

            msg = model.load_state_dict(state_dict, strict=True)
            print(f"[DINOv3-UTAE] Checkpoint loaded successfully ({msg}).")
        else:
            if pretrained:
                print(f"[DINOv3-UTAE] Warning: Checkpoint not found. Initialized with random weights.")
            model.init_weights()

        return model

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: Optional[torch.Tensor] = None,
        return_att: bool = False,
        return_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of DINOv3-UTAE.

        Args:
            x (Tensor): Input tensor of shape (B, T, 17, 128, 128) or (B, 17, 128, 128).
            batch_positions (Tensor, optional): Temporal day indices of shape (B, T).
                If None, defaults to relative time step positions [1, 2, ..., T].
            return_att (bool): If True, returns (output, attention_mask).
            return_maps (bool): If True, returns (output, list_of_decoder_maps).

        Returns:
            Tensor: Prediction logits map of shape (B, 1, 128, 128)
            (optional): Temporal attention weights of shape (n_head, B, T, 8, 8)
        """
        # Handle 4D single-frame input: (B, C, H, W) -> (B, 1, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert H == 128 and W == 128, f"Expected 128x128 spatial resolution, got {H}x{W}"

        # Temporal padding mask: detect dates that are all padded
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # (B, T)

        # Default relative temporal position encoding [1, ..., T] for wildfire prediction
        if batch_positions is None:
            batch_positions = (
                torch.arange(1, T + 1, device=x.device)
                .unsqueeze(0)
                .expand(B, -1)
            )

        # Flatten (B, T) to combined batch dimension for spatial processing
        x_flat = x.view(B * T, C, H, W)

        # ---------------------------------------------------------------------
        # Step 1: High-Resolution Spatial CNN Stem
        # ---------------------------------------------------------------------
        s0 = self.stem0(x_flat)  # (B*T, 64, 128, 128)
        s1 = self.down1(s0)     # (B*T, 64, 64, 64)
        s2 = self.down2(s1)     # (B*T, 128, 32, 32)
        s3 = self.down3(s2)     # (B*T, 256, 16, 16)
        s4 = self.down4(s3)     # (B*T, 384, 8, 8)

        # ---------------------------------------------------------------------
        # Step 2: DINOv3 ViT-S/16 Foundation Model Feature Extraction
        # ---------------------------------------------------------------------
        dino_input = self.input_adapter(x_flat)  # (B*T, 3, 128, 128)
        # Extract intermediate layers at depths: block 2, block 5, block 8, block 11
        d_feats = self.dino.get_intermediate_layers(dino_input, n=[2, 5, 8, 11], reshape=True)

        # ---------------------------------------------------------------------
        # Step 3: Multi-Scale Feature Fusion (DINOv3 + CNN Stem)
        # ---------------------------------------------------------------------
        f1 = s1 + self.deconv1(d_feats[0])          # Level 1: 64x64
        f2 = s2 + self.deconv2(d_feats[1])          # Level 2: 32x32
        f3 = s3 + self.deconv3(d_feats[2])          # Level 3: 16x16
        f4 = s4 + self.proj_bottleneck(d_feats[3])  # Level 4: 8x8 (Bottleneck)

        # Reshape to 5D spatio-temporal representations (B, T, C, H, W)
        f0_seq = s0.view(B, T, self.encoder_widths[0], H, W)
        f1_seq = f1.view(B, T, self.encoder_widths[1], H // 2, W // 2)
        f2_seq = f2.view(B, T, self.encoder_widths[2], H // 4, W // 4)
        f3_seq = f3.view(B, T, self.encoder_widths[3], H // 8, W // 8)
        f4_seq = f4.view(B, T, self.encoder_widths[4], H // 16, W // 16)

        # ---------------------------------------------------------------------
        # Step 4: Temporal Attention Fusion with L-TAE at Bottleneck (8x8)
        # ---------------------------------------------------------------------
        bottleneck_fused, att = self.temporal_encoder(
            f4_seq, batch_positions=batch_positions, pad_mask=pad_mask
        )  # bottleneck_fused: (B, 384, 8, 8), att: (n_head, B, T, 8, 8)

        # ---------------------------------------------------------------------
        # Step 5: Temporal Aggregation of Skip Connections
        # ---------------------------------------------------------------------
        skip3 = self.temporal_aggregator(f3_seq, pad_mask=pad_mask, attn_mask=att)  # (B, 256, 16, 16)
        skip2 = self.temporal_aggregator(f2_seq, pad_mask=pad_mask, attn_mask=att)  # (B, 128, 32, 32)
        skip1 = self.temporal_aggregator(f1_seq, pad_mask=pad_mask, attn_mask=att)  # (B, 64, 64, 64)
        skip0 = self.temporal_aggregator(f0_seq, pad_mask=pad_mask, attn_mask=att)  # (B, 64, 128, 128)

        # ---------------------------------------------------------------------
        # Step 6: 4-Stage Spatial Decoder with Multi-Scale Skip Connections
        # ---------------------------------------------------------------------
        dec_maps = []
        d0 = self.up0(bottleneck_fused, skip3)  # 8x8 -> 16x16 (256 channels)
        if return_maps:
            dec_maps.append(d0)

        d1 = self.up1(d0, skip2)                # 16x16 -> 32x32 (128 channels)
        if return_maps:
            dec_maps.append(d1)

        d2 = self.up2(d1, skip1)                # 32x32 -> 64x64 (64 channels)
        if return_maps:
            dec_maps.append(d2)

        d3 = self.up3(d2, skip0)                # 64x64 -> 128x128 (32 channels)
        if return_maps:
            dec_maps.append(d3)

        # Output prediction map: (B, 1, 128, 128)
        out = self.out_conv(d3)

        if return_att:
            return out, att
        if return_maps:
            return out, dec_maps
        return out


# Aliases for convenience
DINOv3UTAE = DINOv3_UTAE
dinov3_utae = DINOv3_UTAE


# =============================================================================
# Factory Function
# =============================================================================
def build_dinov3_utae(
    in_channels: int = 17,
    n_classes: int = 1,
    checkpoint_path: Optional[str] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> DINOv3_UTAE:
    """
    Convenience factory to build and initialize DINOv3_UTAE for wildfire prediction.
    """
    return DINOv3_UTAE(
        in_channels=in_channels,
        n_classes=n_classes,
        checkpoint_path=checkpoint_path,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


# =============================================================================
# Quick Smoke Test
# =============================================================================
if __name__ == "__main__":
    print("Testing DINOv3_UTAE...")
    model = build_dinov3_utae(
        in_channels=17,
        n_classes=1,
        checkpoint_path="checkpoints/vits16.pth",
        pretrained=True,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dummy_input = torch.randn(2, 5, 17, 128, 128)
    out, att = model(dummy_input, return_att=True)
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", out.shape)
    print("Attn shape:  ", att.shape)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    print("Smoke test passed successfully!")

