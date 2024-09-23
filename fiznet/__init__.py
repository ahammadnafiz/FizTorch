__version__ = '0.1.0'

from .binarymodel import LogisticClassifier
from .neural_network import NN
from .knn import KNNClassifier


__all__ = [
    'LogisticClassifier',
    'NN',
    'KNNClassifier',
]

def get_version():
    return __version__

def get_available_models():
    return [LogisticClassifier, NN, KNNClassifier]