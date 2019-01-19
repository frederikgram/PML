""" Simple Python implementation of a K-Nearest Neighbors classification model """

import operator

from collections import Counter
from models import base_models
from typing import List, NoReturn


class KNearestNeighbor(base_models.Classifier):
    """ K-nearest neighbors ML model"""

    def __init__(self):
        """ Initialises empty data structure """

        self.data = None

    def fit(self, X: List, y: List) -> NoReturn:
        self.data = zip(X, y)

    def predict(self, X: float, k: int = None):
        """
        :param X: data to predict a label for as a float
        :param k: How many neighbors to look at
        :return: Predicted label for X
        """

        # Ensure that model.fit has been executed
        if self.data is None:
            raise ValueError("model needs to be fit before predictions can be made, see model.fit(X, y)")

        # Set default value for k
        if k is None:
            k = len(set([label[1] for label in self.data])) + 1

        # Sort Neighbors by distance
        neighbors = sorted(self.data, key=lambda a: abs(a - X))[:k]

        # Count the most common label in neighbors
        counted_labels = Counter([label[1] for label in neighbors])

        # Return the label with the most presence in the set of neighbors
        return max(counted_labels.items(), key=operator.itemgetter(1))[0]


