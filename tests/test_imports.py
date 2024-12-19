import pytest

def test_all_imports():
    # Import core
    from fiztorch import Tensor
    
    # Import nn modules
    from fiztorch.nn import Linear, ReLU, Sequential
    import fiztorch.nn.functional as F
    
    # Import optim
    from fiztorch.optim import SGD
    
    # Import utils
    from fiztorch.utils import DataLoader
    
    # Basic tensor test
    t = Tensor([1, 2, 3])
    assert t.data.tolist() == [1, 2, 3]
    
    # Basic nn test
    model = Sequential(
        Linear(2, 3),
        ReLU()
    )
    
    # Basic optimizer test
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Basic dataloader test
    import numpy as np
    loader = DataLoader(np.array([[1, 2], [3, 4]]), np.array([0, 1]), batch_size=1)
    
    assert True  # If we got here, all imports worked

if __name__ == "__main__":
    pytest.main([__file__])