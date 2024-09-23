import numpy as np
from .base_model import BaseModel

class LogisticClassifier(BaseModel):
    def __init__(self, input_dim, lambda_reg=0.01):
        self.W = np.zeros((input_dim, 1))
        self.b = 0
        self.lambda_reg = lambda_reg  # L2 regularization parameter

    def sigmoid(self, z):
        # More numerically stable sigmoid function
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))

    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

    def compute_cost(self, A, Y):
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)) / m
        # Add L2 regularization term
        cost += (self.lambda_reg / (2 * m)) * np.sum(np.square(self.W))
        return cost

    def backward(self, X, A, Y):
        m = X.shape[0]
        dz = A - Y
        dW = np.dot(X.T, dz) / m + (self.lambda_reg / m) * self.W  # Add regularization term
        db = np.sum(dz) / m
        return dW, db

    def update_parameters(self, dW, db, learning_rate):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def train(self, X, Y, num_iterations, learning_rate, early_stopping_rounds=5):
        best_cost = float('inf')
        rounds_without_improvement = 0

        for i in range(num_iterations):
            A = self.forward(X)
            cost = self.compute_cost(A, Y)
            dW, db = self.backward(X, A, Y)
            self.update_parameters(dW, db, learning_rate)

            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

            # Early stopping
            if cost < best_cost:
                best_cost = cost
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement >= early_stopping_rounds:
                print(f"Early stopping at iteration {i}")
                break

    def predict(self, X):
        A = self.forward(X)
        return (A > 0.5).astype(int)