from ..tensor import Tensor


class Layer:
    def __init__(self):
        self.tensors = []

    def forward(self, x):
        return x, lambda delta: delta

    def create(self, tensor):
        t = Tensor(tensor)
        self.tensors.append(t)
        return t

    # gets only called by the network to go through all tensor references of its layers to update the weights and biases
    def update(self, optimizer):
        for tensor in self.tensors: optimizer.update(tensor)
