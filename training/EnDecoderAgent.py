import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from model.EnDeCoder import AutoEnDecoder

class AutoEncoderAgent(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self._model = AutoEncoderAgent()

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)['state_dict']
        self.load_state_dict(state_dict)

    def training_step(self, batch, batch_idx):
        batch = batch / 2000
        pred = self._model(batch)
        loss = F.mse_loss(pred, batch)
        self.log('train_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = batch / 2000
        pred = self._model(batch)
        loss = F.mse_loss(pred, batch)
        self.log('val_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=1e-4)
        
        
