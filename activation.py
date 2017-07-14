
# Activation functions for neural networks and their derivatives


import numpy as np


def sigmoid(x, derivative=False):
    if derivative == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative == True:
        return 1 - (x ** 2)
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative == True:
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(len(x)):
        for k in range(len(x[i])):
            if x[i][k] > 0:
                pass
            else:
                x[i][k] = 0
    return x


def leakingRelu(x, derivative=False):
    if derivative == True:
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    return np.maximum(x, x * 0.01)

def identity(x, derivative=False):
    pass # linear act func


def step(x, derivative=False):
    pass # linear act func


def piecewiseLinear(x, derivative=False):
    pass # linear act func


def complementaryLogLog(x, derivative=False):
    pass # linear act func


def bipolar(x, derivative=False):
    pass # linear act func


def arctan(x, derivative=False):
    if derivative == True:
        return (np.cos(x) ** 2)
    return np.arctan(x)

