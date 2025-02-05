# Only import the core Tensor class at root level
from .tensor import Tensor
from . import nn
from . import optim
from . import utils

# Version of the package
__version__ = '0.1.0'

__all__ = ['Tensor', 'nn', 'optim', 'utils']