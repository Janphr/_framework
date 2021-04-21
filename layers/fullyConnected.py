import numpy as np
from .layer import Layer


class FullyConnected(Layer):
    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        # creates the weight matrix with mean 0 and variance 1
        # TODO manipulating for better results, right now using kaiming initialization (1/sqrt(in_size))
        self.weights = self.create(kwargs.get('w', np.random.randn(in_size, out_size) * np.sqrt(1/in_size)))
        self.bias = self.create(kwargs.get('b', np.zeros(out_size)))
        # self.weights = self.create(kwargs.get('w', np.random.randn(in_size, out_size) - 0.5))
        # self.bias = self.create(kwargs.get('b', np.random.rand(out_size) - 0.5))
        # self.weights = self.create(kwargs.get('w', (np.random.uniform(-1, 1, (in_size, out_size)))))

    # returns output for a given input (x) and backward function
    def forward(self, x):
        def backward(delta):
            self.weights.gradient += x.T @ delta
            self.bias.gradient += delta.sum(axis=0)
            # dy * w^T
            return delta @ self.weights.tensor.T

        # x * w + b
        return x @ self.weights.tensor + self.bias.tensor, backward
