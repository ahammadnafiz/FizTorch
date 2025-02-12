from typing import List, Iterator
from .module import Module
from ..tensor import Tensor

class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    A Sequential module is a container module that can be used to create a neural network by stacking layers sequentially.
    
    Raises:
        TypeError: If any of the provided layers is not an instance of Module
        ValueError: If trying to access an invalid layer index
    """

    def __init__(self, *layers: Module):
        """
        Initializes the Sequential container with the given layers.

        Args:
            *layers (Module): Variable number of modules to be added to the container.

        Raises:
            TypeError: If any of the provided layers is not an instance of Module
        """
        super().__init__()
        self.layers: List[Module] = []
        self._parameter_count = 0  # Counter for unique parameter naming
        
        # Add each layer through the append method to ensure proper validation
        for layer in layers:
            self.append(layer)

    def forward(self, input: Tensor) -> Tensor:
        """
        Defines the computation performed at every call. Passes the input through each layer in sequence.

        Args:
            input (Tensor): The input tensor to the network.

        Returns:
            Tensor: The output tensor after passing through all layers.

        Raises:
            ValueError: If the Sequential container has no layers
        """
        if not self.layers:
            raise ValueError("Cannot perform forward pass with no layers")

        current_input = input
        for layer in self.layers:
            current_input = layer(current_input)
        return current_input

    def parameters(self) -> Iterator[Tensor]:
        """
        Returns an iterator over module parameters.

        Yields:
            Iterator[Tensor]: An iterator over the parameters of each layer.
        """
        for layer in self.layers:
            yield from layer.parameters()

    def __getitem__(self, idx: int) -> Module:
        """
        Gets the layer at the given index.

        Args:
            idx (int): Index of the layer to retrieve.

        Returns:
            Module: The layer at the specified index.

        Raises:
            IndexError: If the index is out of bounds
            TypeError: If the index is not an integer
        """
        if not isinstance(idx, int):
            raise TypeError(f"Layer indices must be integers, not {type(idx).__name__}")
        
        if not -len(self.layers) <= idx < len(self.layers):
            raise IndexError(f"Layer index {idx} is out of range")
            
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

        Raises:
            TypeError: If the provided module is not an instance of Module
        """
        if not isinstance(module, Module):
            raise TypeError(f"Sequential expects Module instances, got {type(module).__name__}")
        
        # Generate a unique parameter name
        param_name = f'layer_{self._parameter_count}'
        self._parameter_count += 1
        
        self.layers.append(module)
        self._parameters[param_name] = module

    def insert(self, index: int, module: Module) -> None:
        """
        Inserts a module at the specified position.

        Args:
            index (int): Index at which to insert the module
            module (Module): The module to insert

        Raises:
            TypeError: If the provided module is not an instance of Module
            IndexError: If the index is out of bounds
        """
        if not isinstance(module, Module):
            raise TypeError(f"Sequential expects Module instances, got {type(module).__name__}")
            
        if not -len(self.layers) <= index <= len(self.layers):
            raise IndexError(f"Insert index {index} is out of range")

        # Generate a unique parameter name
        param_name = f'layer_{self._parameter_count}'
        self._parameter_count += 1
        
        self.layers.insert(index, module)
        self._parameters[param_name] = module

    def remove(self, module: Module) -> None:
        """
        Removes the first occurrence of the specified module.

        Args:
            module (Module): The module to remove

        Raises:
            ValueError: If the module is not found in the container
        """
        try:
            self.layers.remove(module)
            # Remove from parameters by value
            for key, value in list(self._parameters.items()):
                if value is module:
                    del self._parameters[key]
                    break
        except ValueError:
            raise ValueError(f"Module {module} not found in Sequential")

    def clear(self) -> None:
        """
        Removes all modules from the container.
        """
        self.layers.clear()
        self._parameters.clear()
        self._parameter_count = 0

    def __iter__(self) -> Iterator[Module]:
        """
        Returns an iterator over the layers in the container.

        Yields:
            Iterator[Module]: An iterator over the layers.
        """
        return iter(self.layers)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Sequential container.

        Returns:
            str: String representation of the container.
        """
        return f"Sequential(\n  {',\n  '.join([str(layer) for layer in self.layers])}\n)"