class MGDOptimizer():
    def __init__(self, lr=0.01, alpha=0, gamma=0.1):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma

    # updates the weights and biases via gradient decent and erases the gradients afterwards case of the use of += in bw
    def update(self, tensor):
        tensor.m = self.gamma * tensor.m + \
                   self.lr * (tensor.gradient + self.alpha * tensor.tensor / len(tensor.tensor))
        tensor.tensor -= tensor.m
        tensor.gradient.fill(0)


class SGDOptimizer():
    def __init__(self, lr=0.01, alpha=0):
        self.lr = lr
        self.alpha = alpha

    # updates the weights and biases via gradient decent and erases the gradients afterwards case of the use of += in bw
    def update(self, tensor):
        tensor.tensor -= self.lr * (tensor.gradient + self.alpha * tensor.tensor / len(tensor.tensor))
        tensor.gradient.fill(0)


class NAGOptimizer():
    def __init__(self, lr=0.01, alpha=0, gamma=0.1):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma

    # updates the weights and biases via gradient decent and erases the gradients afterwards case of the use of += in bw
    def update(self, tensor):
        look_ahead = tensor.gradient + self.alpha * tensor.tensor / len(tensor.tensor) - self.gamma * tensor.m
        tensor.m = self.gamma * tensor.m + self.lr * look_ahead
        tensor.tensor -= tensor.m
        tensor.gradient.fill(0)


class AdagradOptimizer():
    def __init__(self, lr=0.01, alpha=0, gamma=0.1):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma

    # updates the weights and biases via gradient decent and erases the gradients afterwards case of the use of += in bw
    def update(self, tensor):
        tensor.m = tensor.m ** 2 + tensor.m
        tensor.tensor -= (self.lr / (tensor.m + self.gamma) ** .5) * (tensor.gradient + self.alpha * tensor.tensor / len(tensor.tensor))
        tensor.gradient.fill(0)


class AdamOptimizer():
    def __init__(self, lr=0.01, alpha=0, gamma=1e-8, beta1=0.9, beta2=0.99):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2

    # updates the weights and biases via gradient decent and erases the gradients afterwards case of the use of += in bw
    def update(self, tensor):
        tensor.m = self.beta1 * tensor.m + (1 - self.beta1) * tensor.gradient
        tensor.v = self.beta2 * tensor.v + (1 - self.beta2) * tensor.gradient**2

        m = tensor.m / (1 - self.beta1)
        v = tensor.v / (1 - self.beta2)

        tensor.tensor -= (self.lr / (v + self.gamma)**.5) * (m + self.alpha * tensor.tensor / len(tensor.tensor))
        tensor.gradient.fill(0)