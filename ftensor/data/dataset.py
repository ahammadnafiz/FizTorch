# ftensor/data/dataset.py
from typing import List, Tuple
import numpy as np
from ..core import FTensor

class Dataset:
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[FTensor, FTensor]:
        x, y = self.data[idx]
        return FTensor(x), FTensor(y)
