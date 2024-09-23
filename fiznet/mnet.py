from .base_model import BaseModel
from mtorch.core import MTensor
import numpy as np

class NN(BaseModel):
    def __init__(self, layer_dims, lambda_reg=0.0, decay_rate=0.0, keep_prob=1.0):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.L = len(layer_dims) - 1
        self.lambda_reg = lambda_reg
        self.decay_rate = decay_rate
        self.keep_prob = keep_prob

    def initialize_parameters(self):
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return parameters

    def relu(self, Z):
        return Z._elementwise_ops(MTensor(0), lambda z, _: max(0, z))

    def relu_derivative(self, Z):
        return Z._elementwise_ops(MTensor(0), lambda z, _: z > 0)

    def softmax(self, Z):
        Z_max = MTensor([[max([max(row) for row in Z.data])]])  # Broadcasting
        exp_Z = Z - Z_max
        exp_Z = exp_Z._elementwise_ops(MTensor([[2.718281828459045]]), lambda x, y: x ** y)
        return exp_Z / exp_Z._elementwise_ops(exp_Z, lambda _, col: sum([sum(row) for row in col]))

    def forward_propagation(self, X, is_training=True):
        caches = []
        A = X
        D = None  # Dropout mask
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = W.dot(A_prev) + b
            if l == self.L:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
                if is_training and self.keep_prob < 1.0:
                    D = MTensor.random(A.shape)
                    D = D._elementwise_ops(MTensor(self.keep_prob), lambda x, _: x < self.keep_prob)
                    A = A * D
                    A = A._elementwise_ops(MTensor(self.keep_prob), lambda a, _: a / self.keep_prob)
            cache = (A_prev, W, b, Z, D)
            caches.append(cache)
        return A, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        dZL = AL - Y
        A_prev, WL, bL, ZL, _ = caches[self.L - 1]
        grads[f'dW{self.L}'] = (dZL.dot(A_prev.T)) / m + (self.lambda_reg / m) * WL
        grads[f'db{self.L}'] = dZL._elementwise_ops(MTensor([[1]]), lambda z, _: sum(z)) / m
        dA_prev = WL.T.dot(dZL)
        for l in reversed(range(self.L - 1)):
            A_prev, W, b, Z, D = caches[l]
            dZ = dA_prev * self.relu_derivative(Z)
            if self.keep_prob < 1.0:
                dZ = dZ * D
                dZ = dZ._elementwise_ops(MTensor(self.keep_prob), lambda dz, _: dz / self.keep_prob)
            grads[f'dW{l+1}'] = (dZ.dot(A_prev.T)) / m + (self.lambda_reg / m) * W
            grads[f'db{l+1}'] = dZ._elementwise_ops(MTensor([[1]]), lambda z, _: sum(z)) / m
            dA_prev = W.T.dot(dZ)
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = self.parameters[f'W{l}'] - grads[f'dW{l}']._elementwise_ops(MTensor([[learning_rate]]), lambda g, lr: g * lr)
            self.parameters[f'b{l}'] = self.parameters[f'b{l}'] - grads[f'db{l}']._elementwise_ops(MTensor([[learning_rate]]), lambda g, lr: g * lr)

    def train(self, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            AL, caches = self.forward_propagation(X)
            cost = -sum([sum(Y[i]._elementwise_ops(MTensor([[AL[i]]]), lambda y, a: y * MTensor.log(a))) for i in range(len(Y))]) / Y.shape[1]
            l2_cost = sum([W._elementwise_ops(W, lambda w, _: sum([sum(row) for row in w])).data[0][0] for W in self.parameters.values() if 'W' in W])
            cost += (self.lambda_reg / (2 * Y.shape[1])) * l2_cost
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            learning_rate *= (1. / (1. + self.decay_rate * i))
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X, is_training=False)
        return np.argmax(AL, axis=0)
