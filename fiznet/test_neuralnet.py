from ftensor.core.ftensor import FTensor
import math

class NN:
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
            # Initialize weights with He initialization
            W = FTensor([[math.sqrt(2. / self.layer_dims[l-1]) * (2 * math.random() - 1) for _ in range(self.layer_dims[l-1])] for _ in range(self.layer_dims[l])])
            b = FTensor([[0] for _ in range(self.layer_dims[l])])
            parameters[f'W{l}'] = W
            parameters[f'b{l}'] = b
        return parameters

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
                A = Z.softmax()
            else:
                A = Z.relu()
                if is_training and self.keep_prob < 1.0:
                    D = FTensor([[1 if math.random() < self.keep_prob else 0 for _ in range(A.shape[1])] for _ in range(A.shape[0])])
                    A = A * D
                    A = A * (1 / self.keep_prob)
            cache = (A_prev, W, b, Z, D)
            caches.append(cache)
        return A, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dZL = AL - Y
        A_prev, WL, bL, ZL, _ = caches[self.L - 1]
        grads[f'dW{self.L}'] = (dZL.dot(A_prev.transpose()) * (1/m)) + (WL * (self.lambda_reg / m))
        grads[f'db{self.L}'] = dZL.sum(axis=1).reshape(bL.shape) * (1/m)
        dA_prev = WL.transpose().dot(dZL)
        for l in reversed(range(self.L - 1)):
            A_prev, W, b, Z, D = caches[l]
            dZ = dA_prev * Z.relu_derivative()
            if self.keep_prob < 1.0:
                dZ = dZ * D
                dZ = dZ * (1 / self.keep_prob)
            grads[f'dW{l+1}'] = (dZ.dot(A_prev.transpose()) * (1/m)) + (W * (self.lambda_reg / m))
            grads[f'db{l+1}'] = dZ.sum(axis=1).reshape(b.shape) * (1/m)
            if l > 0:
                dA_prev = W.transpose().dot(dZ)
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = self.parameters[f'W{l}'] - grads[f'dW{l}'] * learning_rate
            self.parameters[f'b{l}'] = self.parameters[f'b{l}'] - grads[f'db{l}'] * learning_rate

    def train(self, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            AL, caches = self.forward_propagation(X)
            epsilon = 1e-8
            cost = -(Y * (AL + epsilon).log()).sum() / Y.shape[1]
            # Add L2 regularization term to the cost
            l2_cost = FTensor([0])
            for l in range(1, self.L + 1):
                l2_cost = l2_cost + (self.parameters[f'W{l}'] * self.parameters[f'W{l}']).sum()
            cost = cost + (self.lambda_reg / (2 * Y.shape[1])) * l2_cost
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            # Decay the learning rate
            learning_rate *= (1. / (1. + self.decay_rate * i))
            if i % 100 == 0:
                costs.append(cost.data[0][0])
                print(f"Cost after iteration {i}: {cost.data[0][0]}")
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X, is_training=False)
        return FTensor([AL.data[i].index(max(AL.data[i])) for i in range(len(AL.data))])