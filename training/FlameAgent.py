import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from model.EnDeCoder import AutoEnDecoder
from model.ViTDecoder import ViTDecoder
from diffusers import DDPMScheduler

class FlameAgent(pl.LightningModule):
    def __init__(self, endecoder_ckpt: str):
        '''
        FlameDiffModel, AutoEncoder + ViTDecoder.
        Args:
            endecoder_ckpt: str, path to the checkpoint of AutoEnDecoder.
        '''
        super().__init__()
        self.autoendecoder = AutoEnDecoder()
        self.autoendecoder.load_state_dict(torch.load(endecoder_ckpt))
        self.autoendecoder.eval()
        self.autoendecoder.requires_grad_(False)
        self.vitdecoder = ViTDecoder()
        self.scheduler = DDPMScheduler()


    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def _step(self, batch, batch_idx):
        '''
        Args:
            batch: torch.Tensor, (B, 1, 192, 256)
        '''
        img = batch / 2000 # (B, 1, 192, 256)
        feature = img[:, :, 0::64, 0::64] # Sample 3 * 4 points from the image (B, 1, 3, 4)
        feature = feature.squeeze(1).reshape(feature.shape[0], -1) # (B, 1, 3, 4) -> (B, 12)
        img = self.autoendecoder.encoder(img) # (B, 1, 192, 256) -> (B, 1, 24, 32)
        # add standard normal noise
        noise = torch.randn_like(img).to(img.device)
        timesteps = torch.randint(0, 1000, (batch.shape[0], 1)).to(img.device) # (B, 1)
        noisy_img = self.scheduler.add_noise(img, noise, timesteps)
        pred_noise = self.vitdecoder(noisy_img, feature, timesteps.float())

        loss = F.mse_loss(pred_noise, noise)
        self.log('train_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.vitdecoder.parameters(), lr=1e-4)
        
        
