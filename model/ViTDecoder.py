import torch
import torch.nn as nn

class ViTDecoder(nn.Module):
    def __init__(self, d_model=256, img_size=(24, 32), patch_size=7, num_heads=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_patches = 6 * 8

        # Patch embedding
        # self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size) # (B, 1, 45, 61) -> (B, d_model, 6, 8)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=2, stride=2),  # (B, 1, 45, 61) -> (B, d_model, 22, 30)
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=2, stride=2),  # (B, d_model, 22, 30) -> (B, d_model, 11, 15)
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=2, stride=2),  # (B, d_model, 11, 15) -> (B, d_model, 5, 7)
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=2, padding=1),  # (B, d_model, 5, 7) -> (B, d_model, 6, 8)
        )

        # Feature embedding
        self.feat_embed = nn.Sequential(
            nn.Linear(12, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model)) # (1, 48 + 1, d_model), 1 for feature token

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Upsampling from patch-level to full image (6x8 -> 45x61)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=2, stride=2),  # (B, 256, 6, 8) -> (B, 32, 12, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),  # (B, 32, 12, 16) -> (B, 1, 24, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (B, 1, 24, 32) -> (B, 1, 48, 64)
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=4, padding=0)  # (B, 1, 48, 64) -> (B, 1, 45, 61)
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
        x = self.patch_embed(x) # (B, 1, 45, 61) -> (B, d_model, 6, 8)
        x = x.permute(0, 2, 3, 1).reshape(B, self.num_patches, self.d_model) # (B, d_model, 6, 8) -> (B, 6, 8, d_model) -> (B, 48, d_model)

        # Time & Feature embeddings
        t_emb = self.time_embed(t).unsqueeze(1) # (B, 1, d_model)
        f_emb = self.feat_embed(f).unsqueeze(1) # (B, 1, d_model)

        # Expand noise query to batch size

        # Concatenate
        x = torch.cat([x, f_emb], dim=1) # (B, 48 + 1, d_model)

        # Add positional embedding
        x = x + self.pos_embed + t_emb

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, 48 + 1, d_model)

        # Remove feature token
        x = x[:, 0:-1, :].permute(0, 2, 1).reshape(B, self.d_model, 6, 8)

        # Upsampling to full resolution
        noise_map = self.upsample(x)  # (B, 1, 6, 8) -> (B, 1, 45, 61)

        return noise_map
