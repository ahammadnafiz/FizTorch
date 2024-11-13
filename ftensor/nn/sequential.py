from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def add_module(self, module):
        """Add a new module to the sequential container"""
        self.layers.append(module)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def train(self, mode=True):
        """Sets the module in training mode"""
        super().train(mode)
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train(mode)
        return self