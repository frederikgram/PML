import numpy as np


def euclidian(a: float, b: float) -> float:
    """ Calculates the euclidian distance between floats"""

    return np.linalg.norm(a - b)


distance_function_table = {"euclidian": euclidian}
