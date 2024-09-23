import numpy as np
from collections import Counter
from .base_model import BaseModel

class KNNClassifier(BaseModel):
    def __init__(self, k_neighbors=3):
        self.k_neighbors = k_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def get_distance(self, a, b):
        return np.linalg.norm(a - b)

    def get_k_neighbors(self, X_test_single):
        distances = [(i, self.get_distance(self.X_train[i], X_test_single)) for i in range(len(self.X_train))]
        distances.sort(key=lambda x: x[1])
        return distances[:self.k_neighbors]

    def predict(self, X_test):
        predictions = []
        for X_test_single in X_test:
            k_neighbors = self.get_k_neighbors(X_test_single)
            k_labels = [self.y_train[i] for i, _ in k_neighbors]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions