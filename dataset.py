import os
from typing import Union


import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class FlameDiffDataset(Dataset):
    def __init__(self, cache_path: str):
        '''
        Dataset class for FlameDiff dataset.
        Args:
            cache_path (str): Path to the cache file
        '''
        super().__init__()
        self._cache_path: str = cache_path
        self._data: Union[None, np.ndarray] = None
        self._load_cache()
        
    def _load_cache(self):
        for dirpath, dirnames, filenames in tqdm(os.walk(self._cache_path), desc='Loading cache...', ):
            for file in filenames:
                if file == 'T.npy':
                    data = np.load(os.path.join(dirpath, file))
                    if self._data is None:
                        self._data = data
                    else:
                        self._data = np.concatenate((self._data, data), axis=0)
        print(f'Data loaded. Shape: {self._data.shape}. Min: {self._data.min()}. Max: {self._data.max()}')           

    def max_min(self):
        return self._data.max(), self._data.min()
    
    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self._data[idx]).float().unsqueeze(0)
    