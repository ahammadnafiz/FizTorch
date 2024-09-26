# ftensor/optim/adam.py
from .optimizer import Optimizer
from ..core import FTensor
from typing import List
import numpy as np

class Adam(Optimizer):
    def __init__(self, parameters: List[FTensor], learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)