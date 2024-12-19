# First import the base module
from .module import Module

# Then import implementations that depend on Module
from .layers import Linear, ReLU
from .sequential import Sequential

# Import independent modules
from . import functional

__all__ = [
    'Module',
    'Linear',
    'ReLU',
    'Sequential',
    'functional',
]