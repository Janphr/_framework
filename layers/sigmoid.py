from .layer import Layer
import numpy as np

np.seterr(over='ignore')


class Sigmoid(Layer):
    def forward(self, x):
        sm = 1.0 / (1 + np.exp(-x))

        def backward(delta):
            return delta * sm * (1 - sm)

        return sm, backward

    def __str__(self):
        return 'sm'
