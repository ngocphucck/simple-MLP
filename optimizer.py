import numpy as np


class SGD(object):
    def __init__(self, parameters, grads, lr):
        self.parameters = parameters
        self.grads = grads
        self.lr = lr

    def zero_grad(self):
        for key in self.grads.keys():
            self.grads[key] = np.zeros(self.grads[key].shape)

    def step(self):
        for key in self.parameters.keys():
            self.parameters[key] = self.parameters[key] - self.lr * self.grads["d" + str(key)]
