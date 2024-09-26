# ftensor/data/dataloader.py
from .dataset import Dataset
from ..core import FTensor
from typing import List, Tuple
import numpy as np

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
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

    def _collate_batch(self, batch: List[Tuple[FTensor, FTensor]]) -> Tuple[FTensor, FTensor]:
        x = FTensor(np.stack([item[0].data for item in batch]))
        y = FTensor(np.stack([item[1].data for item in batch]))
        return x, y