
import math
from numpy import *


def compute_error(b, m, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x) + b)**2
    return totalError / float(len(points))


def solve(x):
    return x * m + b


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, n_iterations):
    b = starting_b
    m = starting_m

    for i in xrange(n_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - (m_current * x) + b_current)
        m_gradient += -(2 / N) * x * (y - (m_current * x) + b_current)
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

# ===================================================
# Program Start

if __name__ == "__main__":

    path = "./data.csv"
    points = genfromtxt(path, delimiter=',')

    init_b = 0.0
    init_m = 0.0
    n_iterations = 1000
    learning_rate = 0.0001

    [b, m] = gradient_descent_runner(points, init_b, init_m, learning_rate, n_iterations)

    print b
    print m

