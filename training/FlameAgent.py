import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np

from model.EnDeCoder import AutoEnDecoder
from model.ViTDecoder import ViTDecoder
from diffusers import DDPMScheduler

class FlameAgent(pl.LightningModule):
    def __init__(self, endecoder_ckpt: str, max_v: float = None, min_v: float = None, sensors: list = None, ckpt_path: str = None):
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
        self.max_v = max_v
        self.min_v = min_v
        self.sensors = np.array(sensors) # (8, 2)
        
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)["state_dict"])
        else:
            print("No checkpoint path is provided. Load the model from scratch.")


    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
        
    def _step(self, batch, batch_idx):
        '''
        Args:
            batch: torch.Tensor, (B, 1, 192, 256)
        '''
        if (self.max_v is not None) and (self.min_v is not None):
            batch = (batch - self.min_v) / (self.max_v - self.min_v)
        feature = self._sample_feature(batch) # (B, 8)
        img = self.autoendecoder.encoder(batch) # (B, 1, 192, 256) -> (B, 1, 48, 64)
        # add standard normal noise
        noise = torch.randn_like(img).to(img.device)
        timesteps = torch.randint(0, 1000, (batch.shape[0], 1)).to(img.device) # (B, 1)
        noisy_img = self.scheduler.add_noise(img, noise, timesteps)
        pred_noise = self.vitdecoder(noisy_img, feature, timesteps.float())

        loss = F.mse_loss(pred_noise, noise)
        if self.training:
            self.log('train_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        else:
            self.log('val_loss', loss, prog_bar = True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # gradient clip
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.vitdecoder.parameters(), lr=1e-5)
        
    def configure_callbacks(self):
        from pytorch_lightning.callbacks import ModelCheckpoint, Callback
        checkpoint_callback = ModelCheckpoint(
            filename='flame-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            save_top_k=-1,
            mode='min',
            every_n_epochs=50
        )
        class GenerateCallback(Callback):

            def on_validation_epoch_end(self, trainer, pl_module):
                import matplotlib.pyplot as plt
                batch = next(iter(trainer.val_dataloaders))
                ori_img = batch.detach().cpu().numpy()[:, 0, ...]
                img = pl_module.generate(batch)
                img = img.detach().cpu().numpy()[:, 0, ...]
                from utils import visualize
                fig = visualize(ori_img[0], img[0])
                # get uuid from trainer.logger.log_dir
                pl_module.logger.experiment.add_figure(f'val_img_{trainer.current_epoch}', fig, global_step=trainer.current_epoch)
                plt.close(fig)
                
        generate_callback = GenerateCallback()
        return [checkpoint_callback, generate_callback]
    
    def _sample_feature(self, batch: torch.Tensor):
        '''
        Args:
            batch: torch.Tensor, (B, 1, 192, 256)
        Returns:
            torch.Tensor, (B, 1, 3, 4)
        '''
        _, _, H, W = batch.shape
        # self.sensors (8, 2)
        # swap the last dimension
        idx = self.sensors[:, [1, 0]] # (8, 2)
        idx = torch.tensor(idx).to(batch.device).long() # (8, 2)
        idx_row = idx[:, 0] # (8, )
        idx_col = idx[:, 1] # (8, )
        feature = batch[:, :, idx_row, idx_col].squeeze(1).reshape(batch.shape[0], -1) # (B, 8)
        return feature
    
    def generate(self, batch: torch.Tensor):
        '''
        Args:
            batch: torch.Tensor, (B, 1, 192, 256)
        Returns:
            torch.Tensor, (B, 1, 192, 256)
        '''
        self.eval()
        if torch.cuda.is_available():
            batch = batch.cuda()
            self.cuda()
        if (self.max_v is not None) and (self.min_v is not None):
            batch = (batch - self.min_v) / (self.max_v - self.min_v)
        feature = self._sample_feature(batch) # (B, 8)
        img = self.autoendecoder.encoder(batch) # (B, 1, 192, 256) -> (B, 1, 24, 32)
        noisy_img = torch.randn_like(img).to(img.device)
        timesteps = self.scheduler.timesteps.squeeze() # (1000, )
        # noisy_img = self.scheduler.add_noise(img, noise, timesteps[:, 0])     
        from tqdm import tqdm
        with torch.no_grad():
            for i in tqdm(timesteps):
                timestep = torch.ones((batch.shape[0], 1)).to(img.device).float() * i.float()
                pred_noise = self.vitdecoder(noisy_img, feature, timestep.float())
                res_dict = self.scheduler.step(pred_noise, i.int(), noisy_img)
                noisy_img = res_dict['prev_sample']
        # reverse the normalization  
        img = self.autoendecoder.decoder(noisy_img)
        if (self.max_v is not None) and (self.min_v is not None):
            img = img * (self.max_v - self.min_v) + self.min_v
        return img
