from .layers.softmax import softmax
import numpy as np

# return loss and loss derivative

# mean squared error
def mse_loss(p, t):
    diff = p - t.reshape(p.shape)
    return np.square(diff).mean(), 2 * diff / len(diff)


def bce_loss(p, t):
    L = -t*np.log(p) - (1-t)*np.log(1-p)
    dL = -t/p + (1-t)/(1-p)
    return L.sum(axis=1).mean(), dL / len(t)


def bce_loss2(p, t):
    num = np.exp(p)
    den = num.sum(axis=1).reshape(-1, 1)
    prob = num / den
    log_den = np.log(den)
    ce = np.inner(p - log_den, t)
    return ce.mean(), t - prob / len(t)


# softmax + categorical cross entropy
def cce_loss(p, t):
    sm, backward = softmax(p)
    cce = - (t * np.log(sm)).sum(axis=1)
    return cce.mean(), backward(t)
