import torch
from torch import nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    def __init__(self,):
        '''
        AutoEncoder model, downsample (1, 192, 256) ->(1, 24, 32), modified from famous U-Net model.
        https://arxiv.org/pdf/1505.04597
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=0) # (1, 192, 256) -> (64, 190, 254)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=0) # (64, 190, 254) -> (64, 188, 252)
        self.pool1 = nn.MaxPool2d(2, 2) # (64, 188, 252) -> (64, 94, 126)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0) # (64, 94, 126) -> (128, 92, 124)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=0) # (128, 92, 124) -> (128, 90, 122)
        self.pool2 = nn.MaxPool2d(2, 2) # (128, 90, 122) -> (128, 45, 61)
        self.proj1 = nn.Linear(128, 64) # (128, 45, 61) -> (64, 45, 61)
        self.proj2 = nn.Linear(64, 1) # (64, 45, 61) -> (1, 45, 61)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x)) # (1, 192, 256) -> (64, 190, 254)
        x = F.relu(self.conv2(x)) # (64, 190, 254) -> (64, 188, 252)
        x = self.pool1(x) # (64, 188, 252) -> (64, 94, 126)
        x = F.relu(self.conv3(x)) # (64, 94, 126) -> (128, 92, 124)
        x = F.relu(self.conv4(x)) # (128, 92, 124) -> (128, 90, 122)
        x = self.pool2(x) # (128, 90, 122) -> (128, 45, 61)
        x = x.permute(0, 2, 3, 1)
        x = self.proj1(x)
        x = self.proj2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class AutoDecoder(nn.Module):
    def __init__(self,):
        '''
        AutoDecoder model, upsample (1, 45, 61) -> (1, 192, 256), modified from famous U-Net model.
        https://arxiv.org/pdf/1505.04597
        '''
        super().__init__()
        self.proj1 = nn.Linear(1, 64) # (1, 45, 61) -> (64, 45, 61)
        self.proj2 = nn.Linear(64, 128)  # (64, 45, 61) -> (128, 45, 61)
        self.convup1 = nn.ConvTranspose2d(128, 128, 2, 2) # (128, 45, 61) -> (128, 90, 122)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=2) # (128, 90, 122) -> (128, 92, 124)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=2) # (128, 92, 124) -> (64, 94, 126)
        self.convup2 = nn.ConvTranspose2d(64, 64, 2, 2) # (64, 94, 126) -> (64, 188, 252)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=2) # (64, 188, 252) -> (64, 190, 254)
        self.conv8 = nn.Conv2d(64, 1, 3, padding=2) # (64, 190, 254) -> (1, 192, 256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.proj1(x) 
        x = self.proj2(x)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.convup1(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.convup2(x))
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
        
