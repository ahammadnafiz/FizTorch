from typing import Iterator, List, Optional
from ..tensor import Tensor
import numpy as np


class Optimizer:
    """
    Base class for all optimizers.
    
    Provides basic functionality for managing parameters and their gradients.
    All specific optimizer implementations should inherit from this class.
    
    Attributes:
        parameters: List of trainable parameters (Tensor objects) to optimize.
    """
    
    def __init__(self, parameters: Iterator[Tensor]) -> None:
        """
        Initialize the optimizer with parameters to optimize.
        
        Args:
            parameters: Iterator of Tensor objects representing model parameters.
        """
        self.parameters = list(parameters)
    
    def zero_grad(self) -> None:
        """Reset gradients of all parameters to None."""
        for param in self.parameters:
            param.grad = None
            
    def step(self) -> None:
        """Update parameters based on current gradients.
        
        This method should be implemented by all optimizer subclasses.
        
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    
    Implements standard SGD with optional momentum for accelerated training.
    Updates parameters using the formula:
        v = momentum * v - learning_rate * gradient
        parameter += v
    
    Attributes:
        lr: Learning rate for optimization.
        momentum: Momentum factor (default: 0.0).
        velocities: List of velocity arrays for momentum computation.
    """
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01, 
                 momentum: float = 0.0) -> None:
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: Iterator of model parameters to optimize.
            lr: Learning rate (default: 0.01).
            momentum: Momentum factor (default: 0.0).
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities: List[Optional[np.ndarray]] = [None for _ in self.parameters]
        
    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates parameters using standard SGD or SGD with momentum if momentum > 0.
        """
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
    """
    Adam optimizer implementation.
    
    Implements the Adam optimization algorithm as described in:
    'Adam: A Method for Stochastic Optimization' (Kingma & Ba, 2014).
    
    Attributes:
        lr: Learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        eps: Small constant for numerical stability
        t: Current timestep
        m: First moment estimates
        v: Second moment estimates
    """
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8) -> None:
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: Iterator of model parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # Time step
        
        # Initialize moment estimates
        self.m: List[Optional[np.ndarray]] = [None for _ in self.parameters]  # First moment
        self.v: List[Optional[np.ndarray]] = [None for _ in self.parameters]  # Second moment
        
    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates parameters using the Adam optimization algorithm, which adapts
        learning rates for each parameter using estimates of first and second moments
        of the gradients.
        """
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
    """
    RMSprop optimizer implementation.
    
    Implements the RMSprop optimization algorithm, which adapts learning rates
    by dividing by a running average of squared gradients.
    
    Attributes:
        lr: Learning rate
        alpha: Smoothing constant
        eps: Term added to denominator for numerical stability
        square_avg: Running average of squared gradients
    """
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8) -> None:
        """
        Initialize RMSprop optimizer.
        
        Args:
            parameters: Iterator of model parameters to optimize
            lr: Learning rate (default: 0.01)
            alpha: Smoothing constant (default: 0.99)
            eps: Term added to denominator for numerical stability
        """
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg: List[Optional[np.ndarray]] = [None for _ in self.parameters]
        
    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates parameters using the RMSprop algorithm, which maintains a moving
        average of squared gradients and divides the gradient by the root of this
        average.
        """
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
    """
    Adagrad optimizer implementation.
    
    Implements the Adagrad optimization algorithm, which adapts learning rates
    by scaling them inversely proportional to the square root of the sum of all
    past squared gradients.
    
    Attributes:
        lr: Learning rate
        eps: Term added to denominator for numerical stability
        sum_squares: Sum of squared gradients for each parameter
    """
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01,
                 eps: float = 1e-8) -> None:
        """
        Initialize Adagrad optimizer.
        
        Args:
            parameters: Iterator of model parameters to optimize
            lr: Learning rate (default: 0.01)
            eps: Term added to denominator for numerical stability
        """
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        self.sum_squares: List[Optional[np.ndarray]] = [None for _ in self.parameters]
        
    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates parameters using the Adagrad algorithm, which adapts the learning
        rate for each parameter based on the historical gradient information.
        """
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