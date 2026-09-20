# =============================================================================
# model.py — DDPM U-Net Architecture
# =============================================================================
# This file defines all neural network components used in the DDPM model:
#   1. SinusoidalPositionEmbeddings  — encodes diffusion timestep as a vector
#   2. ConvBlock                     — residual conv block conditioned on time
#   3. DownBlock                     — encoder block with spatial downsampling
#   4. UpBlock                       — decoder block with spatial upsampling
#   5. SmallUNet                     — full U-Net that predicts added noise
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Step 1: Sinusoidal Timestep Embedding
# -----------------------------------------------------------------------------
# Converts a scalar timestep t into a fixed-length embedding vector using
# sinusoidal positional encoding (same idea as in Transformers).
# This lets the network distinguish different noise levels during training.
# -----------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: (B,)  — integer timestep index for each sample in the batch
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        # Compute frequency bands
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -factor)
        # Outer product: (B, half_dim)
        emb = time[:, None].float() * emb[None, :]
        # Concatenate sin and cos → (B, dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# -----------------------------------------------------------------------------
# Step 2: Residual Convolutional Block (time-conditioned)
# -----------------------------------------------------------------------------
# A two-layer conv block with GroupNorm + SiLU activations.
# The timestep embedding is injected via a learned linear projection,
# added to the intermediate feature map (FiLM-style conditioning).
# A residual (skip) connection is included for stable training.
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        # Two 3×3 conv layers
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # Group normalisation (8 groups)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        # Project time embedding to channel dimension for conditioning
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        # 1×1 conv to match channel dims for the residual shortcut
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First conv + norm + activation
        h = F.silu(self.norm1(self.conv1(x)))
        # Inject time embedding: broadcast over spatial dims (H, W)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        # Second conv + norm + activation
        h = F.silu(self.norm2(self.conv2(h)))
        # Residual shortcut
        return h + self.residual(x)


# -----------------------------------------------------------------------------
# Step 3: Encoder Block (DownBlock)
# -----------------------------------------------------------------------------
# Applies a ConvBlock, then spatially downsamples via MaxPool2d (stride 2).
# Returns BOTH the pre-pool feature map (for the skip connection) and the
# downsampled output that continues deeper into the network.
# -----------------------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, time_emb_dim)
        self.pool  = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # h  — full-resolution feature map kept for the skip connection
        # down — halved-resolution map passed to the next encoder stage
        h    = self.block(x, t_emb)
        down = self.pool(h)
        return h, down          # (skip, x_next)


# -----------------------------------------------------------------------------
# Step 4: Decoder Block (UpBlock)
# -----------------------------------------------------------------------------
# Upsamples the feature map with a transposed convolution, concatenates the
# corresponding skip connection from the encoder, then applies a ConvBlock.
# A small nearest-neighbour interpolation guards against off-by-one spatial
# mismatches that occur with 28×28 inputs (odd spatial dimensions after pool).
# -----------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        # 2× transposed conv for upsampling
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After cat with skip: (out_ch + skip_ch) input channels
        self.block = ConvBlock(out_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial size with skip connection if there is a 1-pixel mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        # Concatenate skip connection along channel dim
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


# -----------------------------------------------------------------------------
# Step 5: Small U-Net (noise prediction network)
# -----------------------------------------------------------------------------
# Architecture overview (for 28×28 single-channel images):
#
#   Input (1, 28, 28)
#     └─ init_conv  →  (32, 28, 28)
#         ├─ down1  →  skip1 (64, 28, 28)  |  (64, 14, 14)
#         │   └─ down2  →  skip2 (128, 14, 14)  |  (128, 7, 7)
#         │       └─ mid1 → mid2  →  (128, 7, 7)
#         │   └─ up1  (uses skip2)  →  (64, 14, 14)
#         └─ up2  (uses skip1)  →  (32, 28, 28)
#     └─ final_block → final_conv  →  (1, 28, 28)   ← predicted noise ε
#
# The timestep t is encoded once and reused at every block.
# -----------------------------------------------------------------------------
class SmallUNet(nn.Module):
    def __init__(
        self,
        in_channels:  int = 1,
        base_channels: int = 32,
        time_emb_dim:  int = 128,
    ):
        super().__init__()

        # --- Timestep embedding MLP ---
        # Sinusoidal embedding → two linear layers with SiLU
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Initial projection: map input channels to base_channels ---
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder (downsampling path) ---
        self.down1 = DownBlock(base_channels,     base_channels * 2, time_emb_dim)  # 28 → 14
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 14 → 7

        # --- Bottleneck ---
        self.mid1 = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # --- Decoder (upsampling path) ---
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)  # 7 → 14
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels,     time_emb_dim)  # 14 → 28

        # --- Output head ---
        self.final_block = ConvBlock(base_channels, base_channels, time_emb_dim)
        self.final_conv  = nn.Conv2d(base_channels, in_channels, kernel_size=1)   # 1×1 conv → ε̂

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : noisy image  (B, C, H, W)
            t : timestep     (B,)  — integer indices in [0, T)
        Returns:
            Predicted noise ε̂ with the same shape as x.
        """
        # Encode timestep into a shared embedding for all blocks
        t_emb = self.time_mlp(t)                        # (B, time_emb_dim)

        # Initial feature projection
        x = self.init_conv(x)                           # (B, 32, 28, 28)

        # Encoder — collect skip connections
        skip1, x = self.down1(x, t_emb)                # skip1: (B, 64, 28, 28)
        skip2, x = self.down2(x, t_emb)                # skip2: (B, 128, 14, 14)

        # Bottleneck — two ConvBlocks at the lowest resolution
        x = self.mid1(x, t_emb)                        # (B, 128, 7, 7)
        x = self.mid2(x, t_emb)

        # Decoder — upsample and fuse with skip connections
        x = self.up1(x, skip2, t_emb)                  # (B, 64, 14, 14)
        x = self.up2(x, skip1, t_emb)                  # (B, 32, 28, 28)

        # Final refinement + 1×1 projection to output channels
        x = self.final_block(x, t_emb)
        return self.final_conv(x)                       # (B, 1, 28, 28)  — predicted ε


# -----------------------------------------------------------------------------
# Step 6: Self-Attention Block (spatial)
# -----------------------------------------------------------------------------
# Multi-head self-attention over the H*W spatial positions of a feature map,
# with a residual connection. Applied only at low resolutions (e.g. 16x16,
# 8x8) where the token count is small, this gives the network a GLOBAL view so
# it can enforce coherent layout — "one face per image, eyes/nose/mouth in the
# right place" — which a purely convolutional U-Net with a small receptive
# field struggles to do on larger (128px) images.
# -----------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)   # project to Q, K, V
        self.proj = nn.Conv2d(channels, channels, 1)       # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)                            # each (B, heads, Ch, HW)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhci,bhcj->bhij", q * scale, k).softmax(dim=-1)
        out  = torch.einsum("bhij,bhcj->bhci", attn, v)    # (B, heads, Ch, HW)
        out  = out.reshape(B, C, H, W)
        return x + self.proj(out)                          # residual add


# -----------------------------------------------------------------------------
# Step 7: Configurable U-Net with attention (for larger / RGB images)
# -----------------------------------------------------------------------------
# A deeper, parameterizable cousin of SmallUNet. The number of resolution
# levels is set by `channel_mults` (one multiplier per level), so for a 128px
# input with channel_mults=(1,2,4,8) the spatial size walks
# 128 -> 64 -> 32 -> 16 -> 8 at the bottleneck — far more downsampling than
# SmallUNet's two levels, giving the receptive field needed for whole faces.
# Self-attention is inserted at the resolutions listed in `attn_resolutions`
# (and at the bottleneck) for global coherence.
#
# SmallUNet is intentionally left untouched: MNIST keeps using it by default,
# so old checkpoints and behavior are unaffected. This class is opt-in.
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(
        self,
        in_channels:      int = 3,
        base_channels:    int = 64,
        channel_mults:    tuple = (1, 2, 4, 8),
        time_emb_dim:     int = 256,
        attn_resolutions: tuple = (16, 8),
        image_size:       int = 128,
        num_heads:        int = 4,
    ):
        super().__init__()

        # --- Timestep embedding MLP (same recipe as SmallUNet) ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder: one ConvBlock (+optional attention) + MaxPool per level ---
        self.down_blocks = nn.ModuleList()
        self.down_attns  = nn.ModuleList()
        self.pools       = nn.ModuleList()
        skip_chs = []
        in_ch = base_channels
        res   = image_size
        for m in channel_mults:
            out_ch = base_channels * m
            self.down_blocks.append(ConvBlock(in_ch, out_ch, time_emb_dim))
            self.down_attns.append(
                AttentionBlock(out_ch, num_heads) if res in attn_resolutions else nn.Identity()
            )
            self.pools.append(nn.MaxPool2d(2))
            skip_chs.append(out_ch)
            in_ch = out_ch
            res //= 2

        # --- Bottleneck (always gets attention; it is the lowest resolution) ---
        self.mid_block1 = ConvBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn   = AttentionBlock(in_ch, num_heads)
        self.mid_block2 = ConvBlock(in_ch, in_ch, time_emb_dim)

        # --- Decoder: mirror the encoder, fusing skip connections ---
        self.ups       = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attns  = nn.ModuleList()
        for i in reversed(range(len(channel_mults))):
            skip_ch = skip_chs[i]
            out_ch  = base_channels * channel_mults[i - 1] if i > 0 else base_channels
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(out_ch + skip_ch, out_ch, time_emb_dim))
            res_i = image_size // (2 ** i)
            self.up_attns.append(
                AttentionBlock(out_ch, num_heads) if res_i in attn_resolutions else nn.Identity()
            )
            in_ch = out_ch

        self.final_block = ConvBlock(base_channels, base_channels, time_emb_dim)
        self.final_conv  = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        for block, attn, pool in zip(self.down_blocks, self.down_attns, self.pools):
            x = attn(block(x, t_emb))
            skips.append(x)
            x = pool(x)

        x = self.mid_block2(self.mid_attn(self.mid_block1(x, t_emb)), t_emb)

        for up, block, attn in zip(self.ups, self.up_blocks, self.up_attns):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:            # guard odd-size mismatch
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = attn(block(x, t_emb))

        x = self.final_block(x, t_emb)
        return self.final_conv(x)                          # predicted ε, same shape as input
