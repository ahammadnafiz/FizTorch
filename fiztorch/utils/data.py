import numpy as np
from typing import Iterator, Tuple, List

class DataLoader:
    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            yield self.data[batch_indices], self.labels[batch_indices]

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size