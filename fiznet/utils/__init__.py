# Import key functions from tools.py
from .tools import accuracy, precision, recall, f1_score

# Define __all__ to specify what gets imported when someone does "from FizNet.utils import *"
__all__ = ['accuracy', 'precision', 'recall', 'f1_score']
