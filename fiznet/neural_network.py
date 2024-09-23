import numpy as np
from .base_model import BaseModel

class NN(BaseModel):
    def __init__(self, layer_dims, lambda_reg=0.0, decay_rate=0.0, keep_prob=1.0):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.L = len(layer_dims) - 1
        self.lambda_reg = lambda_reg
        self.decay_rate = decay_rate
        self.keep_prob = keep_prob

    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return parameters

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X, is_training=True):
        caches = []
        A = X
        D = None  # Dropout mask
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            if l == self.L:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
                if is_training and self.keep_prob < 1.0:
                    D = np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob
                    A *= D
                    A /= self.keep_prob
            cache = (A_prev, W, b, Z, D)
            caches.append(cache)
        return A, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dZL = AL - Y
        A_prev, WL, bL, ZL, _ = caches[self.L - 1]
        grads[f'dW{self.L}'] = (1/m) * np.dot(dZL, A_prev.T) + (self.lambda_reg / m) * WL
        grads[f'db{self.L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot(WL.T, dZL)
        for l in reversed(range(self.L - 1)):
            A_prev, W, b, Z, D = caches[l]
            dZ = dA_prev * self.relu_derivative(Z)
            if self.keep_prob < 1.0:
                dZ *= D
                dZ /= self.keep_prob
            grads[f'dW{l+1}'] = (1/m) * np.dot(dZ, A_prev.T) + (self.lambda_reg / m) * W
            grads[f'db{l+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 0:
                dA_prev = np.dot(W.T, dZ)
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']

    def train(self, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            AL, caches = self.forward_propagation(X)
            cost = -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]
            # Add L2 regularization term to the cost
            l2_cost = 0
            for l in range(1, self.L + 1):
                l2_cost += np.sum(np.square(self.parameters[f'W{l}']))
            cost += (self.lambda_reg / (2 * Y.shape[1])) * l2_cost
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            # Decay the learning rate
            learning_rate *= (1. / (1. + self.decay_rate * i))
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X, is_training=False)
        return np.argmax(AL, axis=0)