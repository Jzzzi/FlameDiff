import torch
from torch import nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    def __init__(self,):
        '''
        AutoEncoder model, downsample (1, 192, 256) ->(1, 48, 64), modified from famous U-Net model.
        https://arxiv.org/pdf/1505.04597
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1) # (1, 192, 256) -> (64, 192, 256)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2) # (64, 192, 256) -> (64, 192, 256)
        self.pool1 = nn.MaxPool2d(2, 2) # (64, 192, 256) -> (64, 96, 128)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # (64, 96, 128) -> (128, 96, 128)
        self.conv4 = nn.Conv2d(128, 128, 5, padding=2) # (128, 96, 128) -> (128, 96, 128)
        self.pool2 = nn.MaxPool2d(2, 2) # (128, 96, 128) -> (128, 48, 64)
        self.proj1 = nn.Linear(128, 64) # (128, 48, 64) -> (64, 48, 64)
        self.proj2 = nn.Linear(64, 1) # (64, 48, 64) -> (1, 48, 64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.pool1(x)
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 3, 1)
        x = F.selu(self.proj1(x))
        x = F.tanh(self.proj2(x))
        x = x.permute(0, 3, 1, 2)
        return x

class AutoDecoder(nn.Module):
    def __init__(self,):
        '''
        AutoDecoder model, upsample (1, 48, 64) -> (1, 192, 256), modified from famous U-Net model.
        https://arxiv.org/pdf/1505.04597
        '''
        super().__init__()
        self.proj1 = nn.Linear(1, 64) # (1, 48, 64) -> (64, 48, 64)
        self.proj2 = nn.Linear(64, 128)  # (64, 48, 64) -> (128, 48, 64)
        self.conv5 = nn.Conv2d(128, 128, 5, padding=2) # (128, 96, 128) -> (128, 96, 128)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1) # (128, 96, 128) -> (64, 96, 128)
        self.conv7 = nn.Conv2d(64, 64, 5, padding=2) # (64, 192, 256) -> (64, 192, 256)
        self.conv8 = nn.Conv2d(64, 1, 3, padding=1) # (64, 192, 256) -> (1, 192, 256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.selu(self.proj1(x) )
        x = F.selu(self.proj2(x))
        x = x.permute(0, 3, 1, 2) # (1, 48, 64) -> (128, 48, 64)
        x = F.interpolate(x, scale_factor=2) # (128, 48, 64) -> (128, 96, 128)
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))
        x = F.interpolate(x, scale_factor=2) # (128, 96, 128) -> (128, 192, 256)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x
    
class AutoEnDecoder(nn.Module):
    def __init__(self,):
        '''
        AutoEncoderDecoder model, downsample (1, 192, 256) ->(1, 24, 32) -> upsample (1, 192, 256), modified from famous U-Net model.
        https://arxiv.org/pdf/1505.04597
        '''
        super().__init__()
        self.encoder = AutoEncoder()
        self.decoder = AutoDecoder()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x
        x = self.encoder(x)
        x = self.decoder(x)
        x = x
        return x        
        
