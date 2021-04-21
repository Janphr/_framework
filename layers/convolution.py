# import numpy as np
# from project.framework.layers.layer import Layer
#
#
# class Convolution(Layer):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.weights = self.create(np.random.randn(in_size, out_size) * np.sqrt(1 / in_size))
#         self.bias = self.create(np.zeros(out_size))
#
#     def forward(self, x):
#         def backward(delta):
#             self.weights.gradient += x.T @ delta
#             self.bias.gradient += delta.sum(axis=0)
#             # dy * w^T
#             return delta @ self.weights.tensor.T
#
#         # x * w + b
#         return x @ self.weights.tensor + self.bias.tensor, backward
