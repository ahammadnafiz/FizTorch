from typing import List, Iterator
from .module import Module
from ..tensor import Tensor

class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    A Sequential module is a container module that can be used to create a neural network by stacking layers sequentially.
    """

    def __init__(self, *layers: Module):
        """
        Initializes the Sequential container with the given layers.

        Args:
            *layers (Module): Variable number of modules to be added to the container.
        """
        super().__init__()
        self.layers = list(layers)
        for idx, layer in enumerate(self.layers):
            self._parameters[f'layer_{idx}'] = layer  # Store each layer in the parameters dictionary

    def forward(self, input: Tensor) -> Tensor:
        """
        Defines the computation performed at every call. Passes the input through each layer in sequence.

        Args:
            input (Tensor): The input tensor to the network.

        Returns:
            Tensor: The output tensor after passing through all layers.
        """
        for layer in self.layers:
            input = layer(input)  # Pass the input through each layer
        return input

    def parameters(self) -> Iterator[Tensor]:
        """
        Returns an iterator over module parameters.

        Yields:
            Iterator[Tensor]: An iterator over the parameters of each layer.
        """
        for layer in self.layers:
            yield from layer.parameters()  # Yield parameters from each layer

    def __getitem__(self, idx: int) -> Module:
        """
        Gets the layer at the given index.

        Args:
            idx (int): Index of the layer to retrieve.

        Returns:
            Module: The layer at the specified index.
        """
        return self.layers[idx]

    def __len__(self) -> int:
        """
        Returns the number of layers in the container.

        Returns:
            int: The number of layers.
        """
        return len(self.layers)

    def append(self, module: Module) -> None:
        """
        Appends a module to the end of the container.

        Args:
            module (Module): The module to append.
        """
        self.layers.append(module)  # Add the module to the list of layers
        self._parameters[f'layer_{len(self.layers)-1}'] = module  # Store the new layer in the parameters dictionary