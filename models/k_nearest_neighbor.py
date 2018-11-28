from collections import Counter
import operator

from tools import distances
from models import base_models


class k_nearest_neighbor(base_models.Classifier):
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

        # Assert type uniformity in data set
        if not all(isinstance(point, type(X[0])) for point in X):
            raise TypeError("Every value in X has to be of the same type")

        self.data = list(zip(X, y))

    def predict(self, X: list or float, k: int = None, distance_metric: str="euclidian"):
        """

        :param X: List of data
        :param k: List of labels
        :param distance_metric: Type of distance calculation to use
        :return: Predicted label for X
        """

        # Check validity of metric function
        if distance_metric not in distances.distance_function_table:
            raise ValueError("distance metric {metric} is not a supported metric".format(metric=distance_metric))

        # Ensure that model.fit has been executed
        if self.data is None:
            raise ValueError("model needs to be fit before predictions can be made, see model.fit(X, y)")

        # Assert type uniformity in fitted data and data to predict
        if not isinstance(X, type(self.data[0][0])):
            raise TypeError("variable X: {X_type} is not of the same type as variables in the data set {data_type}".format(X_type=type(X), data_type=type(self.data[0][0])))

        if k is None:
            k = len(set([label[1] for label in self.data])) + 1

        try:
            # Find k neighbors sorted by distance from low to high
            neighbors = sorted(self.data, key=lambda a: distances.distance_function_table[distance_metric](a[0], X))[:k]
        except TypeError:
            print("Distance calculations are not supported for type {}".format(type(X)))  # TODO Switch to logging module instead of print
            raise

        # Count the most common label in neighbors
        counted_labels = Counter([label[1] for label in neighbors])
        return max(counted_labels.items(), key=operator.itemgetter(1))[0]


