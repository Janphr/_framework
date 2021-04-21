import numpy as np
from .layer import Layer


class Dropout(Layer):
    # p is the propability for a neuron's output not to be dropped out
    def __init__(self, size, p, **kwargs):
        super().__init__()
        # vector containing zeroes and ones
        self.u = np.random.binomial(1, p, size=size)

    # returns output for a given input (x) and backward function
    def forward(self, x):
        def backward(delta):
            return delta * self.u

        return x * self.u, backward
