import numpy as np


def softmax(tensor):

    return np.exp(-tensor) / np.sum(np.exp(-tensor), axis=0)


def relu(tensor):
    tensor[tensor < 0] = 0

    return tensor


class SimpleMLP(object):
    def __init__(self, hidden_layers):
        self.parameters = {}
        self.caches = {}
        self.grads = {}
        self.hidden_layers = hidden_layers
        self.init_weight(hidden_layers)
        self.init_grads(hidden_layers)

    def forward(self, input_tensor):
        output_tensor = None
        self.caches["A0"] = input_tensor
        for i in range(len(self.hidden_layers) - 1):
            if i == 0:
                output_tensor = np.dot(self.parameters["W1"], input_tensor) + self.parameters["b1"]
            else:
                output_tensor = np.dot(self.parameters["W" + str(i + 1)], output_tensor) + \
                                self.parameters["b" + str(i + 1)]

            if i != len(self.hidden_layers) - 2:
                self.caches["Z" + str(i + 1)] = output_tensor
                output_tensor = relu(output_tensor)
                self.caches["A" + str(i + 1)] = output_tensor

            else:
                self.caches["Z" + str(i + 1)] = output_tensor
                output_tensor = softmax(output_tensor)
                self.caches["A" + str(i + 1)] = output_tensor

        return output_tensor, self.caches, self.grads, self.parameters

    def init_weight(self, hidden_layers):
        for i in range(len(hidden_layers) - 1):
            self.parameters["W" + str(i + 1)] = np.random.randn(hidden_layers[i + 1], hidden_layers[i]) * 0.1
            self.parameters["b" + str(i + 1)] = np.zeros((hidden_layers[i + 1], 1))

    def init_grads(self, hidden_layers):
        for i in range(len(hidden_layers) - 1):
            self.grads["dW" + str(i + 1)] = np.zeros((hidden_layers[i + 1], hidden_layers[i]))
            self.grads["db" + str(i + 1)] = np.zeros((hidden_layers[i + 1], 1))
            self.grads["dZ" + str(i + 1)] = np.zeros((hidden_layers[i + 1], 1))
