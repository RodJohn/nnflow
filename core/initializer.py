import numpy as np


class Initializer(object):

    def init(self, shape):
        raise NotImplementedError


class ZerosInit(Initializer):

    def init(self, shape):
        return np.zeros(shape=shape)


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


class XavierUniformInit(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks”
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))

    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low=-a, high=a, size=shape)
