import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from fiztorch import Tensor

def test_tensor_creation():
    data = [[1, 2], [3, 4]]
    t = Tensor(data)
    assert np.array_equal(t.data, np.array(data))

def test_tensor_addition():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 + t2
    assert np.array_equal(result.data, np.array([5, 7, 9]))

if __name__ == "__main__":
    pytest.main([__file__])