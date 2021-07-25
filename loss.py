import numpy as np


class CrossEntropyLoss(object):
    def __init__(self):
        self.caches = None
        self.grads = None
        self.parameters = None
        self.labels = None

    def forward(self, predict, labels):
        self.caches = predict[1]
        self.grads = predict[2]
        self.parameters = predict[3]
        label_one_hot = np.zeros(predict[0].shape)

        for i in range(labels.shape[0]):
            label_one_hot[labels[i], i] = 1
        self.labels = label_one_hot

        return -1.0 / labels.shape[0] * np.squeeze(np.sum(label_one_hot * np.log(predict[0]) + (1 - label_one_hot) *
                                                          np.log(1 - predict[0])))

    def backward(self):
        for i in range(len(self.grads) // 3, 0, -1):
            if i == len(self.grads) // 3:
                self.grads["dZ" + str(i)] = self.caches["A" + str(i)] - self.labels
            else:
                self.grads["dZ" + str(i)] = np.dot(self.parameters["W" + str(i + 1)].T,
                                                   self.grads["dZ" + str(i + 1)]) * \
                                                  (self.caches["Z" + str(i)] > 0).astype(float)

            self.grads["dW" + str(i)] = 1.0 / self.labels.shape[1] * np.dot(self.grads["dZ" + str(i)],
                                                                            self.caches["A" + str(i - 1)].T)
            self.grads["db" + str(i)] = 1.0 / self.labels.shape[1] * np.sum(self.grads["dZ" + str(i)], axis=1,
                                                                            keepdims=True)
