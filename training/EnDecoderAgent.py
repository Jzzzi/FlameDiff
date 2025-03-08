import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from model.EnDeCoder import AutoEnDecoder

class AutoEncoderAgent(pl.LightningModule):
    def __init__(self, max_v: float = None, min_v: float = None):
        super().__init__()
        self._model = AutoEnDecoder()
        self.max_v = None
        self.min_v = None
        if max_v is not None:
            self.max_v = max_v
        if min_v is not None:
            self.min_v = min_v

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)['state_dict']
        self.load_state_dict(state_dict)

    def training_step(self, batch, batch_idx):
        if (self.max_v is not None) and (self.min_v is not None):
            batch = (batch - self.min_v) / (self.max_v - self.min_v)
        pred = self._model(batch)
        loss = F.mse_loss(pred, batch)
        self.log('train_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if (self.max_v is not None) and (self.min_v is not None):
            batch = (batch - self.min_v) / (self.max_v - self.min_v)
        pred = self._model(batch)
        loss = F.mse_loss(pred, batch)
        self.log('val_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=1e-4)
        
        
