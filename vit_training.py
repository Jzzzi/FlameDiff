import os

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F 
import hydra
from omegaconf import DictConfig

from dataset import FlameDiffDataset
from model.EnDeCoder import AutoEnDecoder
from training.EnDecoderAgent import AutoEncoderAgent
from training.FlameAgent import FlameAgent
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path='config', config_name='default_training', version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    dataset = FlameDiffDataset(cache_path=cfg.cache_path)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
    train_data_loader = DataLoader(train_dataset, **cfg.dataloader)
    val_data_loader = DataLoader(val_dataset, **cfg.dataloader)
    agent = FlameAgent(endecoder_ckpt=cfg.endecoder_ckpt)
    # devices=[0, 2, 3, 4, 5, 6, 7]
    devices=[0, 1, 2, 3, 4, 5, 6, 7]
    # devices = [0]
    trainer = pl.Trainer(**cfg.trainer, devices=devices, logger=TensorBoardLogger(cfg.log_dir, name=cfg.exp_name), default_root_dir=cfg.log_dir)
    trainer.fit(agent,
                train_dataloaders=train_data_loader,
                val_dataloaders=val_data_loader
                )

if __name__ == '__main__':
    main()
