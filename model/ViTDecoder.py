import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000) / (half_dim - 1))).to(timesteps.device) # (d_model // 2)
    emb = timesteps * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb 

class ViTDecoder(nn.Module):
    def __init__(self, d_model=256, img_size=(24, 32), patch_size=7, num_heads=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_patches = 6 * 8

        # Patch embedding
        # self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size) # (B, 1, 45, 61) -> (B, d_model, 6, 8)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model//2, kernel_size=3, stride=2),  # (B, 1, 45, 61) -> (B, d_model, 22, 30)
            nn.ReLU(),
            nn.LayerNorm([d_model//2, 22, 30]),
            nn.Conv2d(d_model//2, d_model, kernel_size=3, stride=2),  # (B, d_model, 22, 30) -> (B, d_model, 10, 14)
        )

        # Feature embedding
        self.feat_embed = nn.Sequential(
            nn.Linear(12, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 10 * 14 + 1, d_model)) # (1, 10 * 15 + 1, d_model), 1 for feature token

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)

        self.transformer_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            nn.LayerNorm(d_model)
        )


        # Upsampling from patch-level to full image (6x8 -> 45x61)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: torch.Tensor, input image, (B, 1, 45, 61)
            f: torch.Tensor, feature, (B, 12)
            t: torch.Tensor, time, (B, 1)
        Returns:
            noise_map: torch.Tensor, predicted noise map, (B, 1, 45, 61)
        '''
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x) # (B, 1, 45, 61) -> (B, d_model, 9, 13)
        x = x.permute(0, 2, 3, 1).reshape(B, 10 * 14, self.d_model).contiguous() # (B, d_model, 10, 15) -> (B, 10, 15, d_model) -> (B, 150, d_model)

        # Time & Feature embeddings
        t_emb = timestep_embedding(t, self.d_model).to(x.device).unsqueeze(1) # (B, 1, d_model)
        f_emb = self.feat_embed(f).unsqueeze(1) # (B, 1, d_model)

        # Concatenate
        x = torch.cat([x, f_emb], dim=1) # (B, 150 + 1, d_model)

        # Add positional embedding
        x = x + self.pos_embed + t_emb

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, 150 + 1, d_model)

        # Remove feature token
        x = x[:, 0:-1, :].permute(0, 2, 1).reshape(B, self.d_model, 10, 14).contiguous()

        # Upsampling to full resolution
        x = self.upsample(x)  # (B, 1, 6, 8) -> (B, 1, 45, 61)
        x = F.interpolate(x, size=(45, 61), mode='bilinear', align_corners=False)  # (B, 1, 6, 8) -> (B, 1, 45, 61)

        return x
