from .layer import Layer
import numpy as np


class Tanh(Layer):
    def forward(self, x):

        def backward(delta):
            return delta * (1 - np.square(np.tanh(x)))

        return np.tanh(x), backward

    def __str__(self):
        return 'th'
