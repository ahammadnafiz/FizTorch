from .tensor import Tensor
import numpy as np

class Function:
    def __init__(self):
        self.parents = []
        self.saved_tensors = []
        self.requires_grad = False
        self.needs_input_grad = []

    @classmethod
    def call(cls, *args, **kwargs):
        # Create context
        ctx = cls()
        
        # Convert inputs to Tensors if necessary
        processed_args = []
        needs_input_grad = []
        
        for arg in args:
            if isinstance(arg, (int, float)):
                processed_args.append(Tensor(arg, requires_grad=False))
                needs_input_grad.append(False)
            elif isinstance(arg, Tensor):
                processed_args.append(arg)
                needs_input_grad.append(arg.requires_grad)
            else:
                raise TypeError(f"Unsupported input type: {type(arg)}")
        
        # Store parents and grad requirements
        ctx.parents = processed_args
        ctx.needs_input_grad = needs_input_grad
        ctx.requires_grad = any(needs_input_grad)
        
        # Forward pass
        outputs = ctx.forward(*processed_args, **kwargs)
        
        # Handle multiple outputs
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
            
        # Wrap outputs in Tensors if necessary
        processed_outputs = []
        for output in outputs:
            if isinstance(output, Tensor):
                output.ctx = ctx if ctx.requires_grad else None
                processed_outputs.append(output)
            else:
                tensor_output = Tensor(output, requires_grad=ctx.requires_grad)
                tensor_output.ctx = ctx if ctx.requires_grad else None
                processed_outputs.append(tensor_output)
        
        return processed_outputs[0] if len(processed_outputs) == 1 else tuple(processed_outputs)

    def save_for_backward(self, *tensors):
        """Save tensors needed for gradient computation"""
        self.saved_tensors = tensors

    def forward(self, *args, **kwargs):
        """
        Forward pass computation
        Args should be Tensor objects
        """
        raise NotImplementedError("Forward pass not implemented")

    def backward(self, grad_output):
        """
        Backward pass computation
        grad_output should be a Tensor object
        Should return a tuple of gradients for each input
        """
        raise NotImplementedError("Backward pass not implemented")

    @staticmethod
    def get_backward_gradient(parent, grad):
        """Helper method to accumulate gradients"""
        if parent.grad is None:
            parent.grad = Tensor(np.zeros_like(parent.data))
        parent.grad.data += grad.data