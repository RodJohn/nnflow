import numpy as np


class BaseLoss(object):
    def loss(self, pred, target):
        raise NotImplementedError

    def grad(self, pred, target):
        raise NotImplementedError


class MSELoss(BaseLoss):
    def loss(self, pred, target):
        n = target.shape[0]
        return 0.5 * np.sum((pred - target) ** 2) / n

    def grad(self,pred,target):
        n = target.shape[0]
        return (pred - target)/n


class SoftmaxCrossEntropyLoss(BaseLoss):

    def loss(self, predicted, actual):
        m = predicted.shape[0]
        # Softmax
        exps = np.exp(predicted - np.max(predicted))
        p = exps / np.sum(exps)
        # cross entropy loss
        nll = -np.log(np.sum(p * actual, axis=1))

        return np.sum(nll) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grad = np.copy(predicted)
        grad -= actual
        return grad / m
