""" Simple but dynamic Perceptron model """

import numpy as np
import random

from typing import NoReturn
from models.base_models import Model


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> float:
    return x * (1 - x)


class Perceptron(Model):
    """ Basic Perceptron with built-in sigmoid trainer """

    def __init__(self, num_features: int):
        self.weights = [random.uniform(-1, 1) for _ in range(num_features)]

    def sigmoid_train(self,  X: list, y: list) -> NoReturn:
        """ Iteratively adjusts weights based on prediction errors """

        for t_x, t_y in zip(X, y):
            output = self.predict(t_x)
            err = t_y - output
            adj = np.dot(t_y, err * sigmoid_derivative(output))
            self.weights += adj

    def predict(self, x: list):
        """ Predicts single output from n len list of integers
            using the sigmoid function
        """

        return sigmoid(np.dot(x, self.weights))


if __name__ == "__main__":

    # Run a basic example of perceptron usage
    # there is not enough training data for this
    # to be accurate, but its an example.

    P = Perceptron(3)
    train_X, train_y = [[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]], [1, 1, 0, 0, 1]

    test_X = [0, 0, 1]
    test_y = [1]

    print(P.predict(test_X))
    P.sigmoid_train(train_X, train_y)
    print(P.predict(test_X))
