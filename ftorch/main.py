# main.py

from core import FTensor

a = FTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = FTensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])


print(a + b)