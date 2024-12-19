# Only import the core Tensor class at root level
from .tensor import Tensor

__version__ = '0.1.0'

# Note: Removed automatic imports of submodules to prevent circular imports