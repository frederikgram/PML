from collections import Counter
import operator

from models import base_models


class KNearestNeighbor(base_models.Classifier):
    """ K-nearest neighbors ML model"""

    def __init__(self):
        """ Initialises empty data structure """

        self.data = None

    def fit(self, X: list, y: list) -> None:
        """

        :param X: List of data
        :param y: List of labels
        :return: No return, sets state of self.data to list(zip(X, y))
        """

        if len(X) != len(y):
            raise ValueError("X and y has to be the same length")

        self.data = list(zip(X, y))

    def predict(self, X: float, k: int = None, distance_metric=lambda a, b: abs(a - b)):
        """
        :param X: List of data
        :param k: List of labels
        :param distance_metric: Type of distance calculation to use
        :return: Predicted label for X
        """

        # Ensure that model.fit has been executed
        if self.data is None:
            raise ValueError("model needs to be fit before predictions can be made, see model.fit(X, y)")

        # Set default value for k
        if k is None:
            k = len(set([label[1] for label in self.data])) + 1

        try:
            # Find k neighbors sorted by distance from low to high
            neighbors = sorted(self.data, key=lambda a: distance_metric(a[0], X))[:k]
        except TypeError:
            print("Distance calculations are not supported for type {}".format(type(X)))
            raise

        # Count the most common label in neighbors
        counted_labels = Counter([label[1] for label in neighbors])
        return max(counted_labels.items(), key=operator.itemgetter(1))[0]


