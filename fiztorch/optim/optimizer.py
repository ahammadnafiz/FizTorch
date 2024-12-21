from typing import Iterator, List, Optional
from ..tensor import Tensor
import numpy as np

class Optimizer:
    def __init__(self, parameters: Iterator[Tensor]):
        self.parameters = list(parameters)
    
    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None
            
    def step(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities: List[Optional[np.ndarray]] = [None for _ in self.parameters]
        
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                if self.momentum > 0:
                    if self.velocities[i] is None:
                        self.velocities[i] = np.zeros_like(param.data)
                    
                    # Update velocity
                    self.velocities[i] = (self.momentum * self.velocities[i] - 
                                        self.lr * param.grad.data)
                    # Update parameters using velocity
                    param.data += self.velocities[i]
                else:
                    # Standard SGD update
                    param.data -= self.lr * param.grad.data

class Adam(Optimizer):
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # Time step
        
        # Initialize moment estimates
        self.m: List[Optional[np.ndarray]] = [None for _ in self.parameters]  # First moment
        self.v: List[Optional[np.ndarray]] = [None for _ in self.parameters]  # Second moment
        
    def step(self) -> None:
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad.data
                
                # Initialize moments if empty
                if self.m[i] is None:
                    self.m[i] = np.zeros_like(param.data)
                    self.v[i] = np.zeros_like(param.data)
                
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grad)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop(Optimizer):
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01, 
                 alpha: float = 0.99, eps: float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg: List[Optional[np.ndarray]] = [None for _ in self.parameters]
        
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad.data
                
                # Initialize square average if empty
                if self.square_avg[i] is None:
                    self.square_avg[i] = np.zeros_like(param.data)
                
                # Update running average of squared gradients
                self.square_avg[i] = (self.alpha * self.square_avg[i] + 
                                    (1 - self.alpha) * np.square(grad))
                
                # Update parameters
                param.data -= (self.lr * grad / 
                             (np.sqrt(self.square_avg[i]) + self.eps))

class Adagrad(Optimizer):
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01, eps: float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        self.sum_squares: List[Optional[np.ndarray]] = [None for _ in self.parameters]
        
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad.data
                
                # Initialize sum of squares if empty
                if self.sum_squares[i] is None:
                    self.sum_squares[i] = np.zeros_like(param.data)
                
                # Accumulate squared gradients
                self.sum_squares[i] += np.square(grad)
                
                # Update parameters
                param.data -= (self.lr * grad / 
                             (np.sqrt(self.sum_squares[i]) + self.eps))