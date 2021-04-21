from .layer import Layer


class ReLU(Layer):
    def forward(self, x):
        mask = x > 0
        return x * mask, lambda delta: delta * mask

    def __str__(self):
        return 'rl'
