import numpy as np


class Tensor():
    def __init__(self, tensor):
        self.tensor = tensor
        self.gradient = np.zeros_like(self.tensor, dtype=float)
        self.m = np.zeros_like(self.tensor, dtype=float)
        self.v = np.zeros_like(self.tensor, dtype=float)


