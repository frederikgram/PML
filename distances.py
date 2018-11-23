import numpy as np


def euclidian(a: float or list, b: float or list) -> float or list:
    """ Calculates the euclidian distance between ints/floats or lists of them"""
    if isinstance(a, list):
        a = sum(a)

    if isinstance(b, list):
         b = sum(b)

    return np.linalg.norm(a - b)


print(euclidian([1, 5], [2,7]))