import numpy as np

from core.initializer import ZerosInit, XavierUniformInit


class Layer(object):

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self,
                 in_num,
                 out_num
                 ):
        super().__init__("linear")
        self.params = {
            "w": XavierUniformInit.init([in_num, out_num]),
            "b": ZerosInit.init([1, in_num])
        }
        self.inputs = None
        self.grad = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grad["w"] = self.inputs.T @ grad
        self.grad["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T


class Activation(Layer):

    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        return self.func(inputs)

    def backward(self, grad):
        self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, grad):
        raise NotImplementedError


class Sigmoid(Activation):

    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_func(self, x):
        return self.func(x) * (1.0 - self.func(x))
