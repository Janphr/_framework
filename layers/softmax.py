import numpy as np


def softmax(x):
    num = np.exp(x - np.max(x))
    sm = num / num.sum(axis=1).reshape(-1, 1)

    def backward(t):
        return (sm - t) / len(t)

    return sm, backward
