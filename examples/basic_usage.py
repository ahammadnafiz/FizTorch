if __name__ == "__main__":
    import sys
    import os
    import numpy as np

    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import the required modules
    from ftensor import nn
    from ftensor import optim
    from ftensor import data
    from ftensor import utils
    from ftensor.core.tensor import Tensor

