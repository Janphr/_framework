from .layers import Layer, softmax
import numpy as np


class Network(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        # create a list of references to the tensors (weights, biases) of all layers
        # to loop through them in the optimizer easier
        for layer in layers:
            self.tensors.extend(layer.tensors)

    def forward(self, x):
        bw_functions = []
        y = x
        # go through each layers, save the backward function and feed the result to the next layer
        for layer in self.layers:
            y, bw_func = layer.forward(y)
            bw_functions.append(bw_func)

        # go through each backward functions in reverse order and feed the gradient to the next bw function
        # first delta is the gradient of the loss function
        def backward(delta):
            for bw_f in reversed(bw_functions):
                delta = bw_f(delta)
            return delta

        return y, backward

    def test(self, x, t):
        tp = 0
        confidence = 0
        classes = len(t[0])
        correct_predictions = [[] for _ in range(classes)]
        wrong_predictions = [[] for _ in range(classes)]
        confusion_matrix = np.zeros((classes, classes))

        for i in range(len(x)):
            y, b = self.forward(x[i])
            p = softmax([y])[0][0]
            t_idx = int(np.argmax(t[i]))
            p_idx = int(np.argmax(p))
            confusion_matrix[t_idx][p_idx] += 1
            if p_idx == t_idx:
                tp += 1
                confidence += p[t_idx]
                correct_predictions[t_idx].append(i)
            else:
                wrong_predictions[t_idx].append(i)
        # return the percentage of true positives and the average confidence
        return tp / len(x), confidence / len(x), [correct_predictions, wrong_predictions], confusion_matrix

    def predict(self, x):
        y, b = self.forward(x)
        p = softmax([y])[0][0]
        # index of the probability >50%
        c = int(np.argmax(p))
        # return index (class) and its probability (confidence)
        return c, p[c]
