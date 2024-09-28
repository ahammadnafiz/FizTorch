from .dataset import Dataset
from ..core import FTensor
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield self._collate_batch([self.dataset[j] for j in batch_indices])

    def _collate_batch(self, batch):
        x = FTensor(np.stack([item[0].data for item in batch]))
        y = FTensor(np.stack([item[1].data for item in batch]))
        return x, y