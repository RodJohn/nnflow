# Author: borgwang <borgwang@126.com>
#
# Filename: BaseOptimizer.py
# Description:
#   Implement multiple optimization algorithms


import numpy as np


class Optimizer:

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads, params):
        # compute step according to derived class method
        grad_values = grads.values
        step_values = self._compute_step(grad_values)
        grads.values = step_values

        # apply weight_decay if specified
        if self.weight_decay:
            grads -= self.lr * self.weight_decay * params

        # take a step
        params += grads

    def _compute_step(self, grad):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def _compute_step(self, grad):
        return -self.lr * grad

