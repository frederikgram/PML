from collections import Counter
import operator

import distances

#  TODO fill out docstrings


class Model:
    """Base class for ML models, this contains various functions with wide use cases"""

    def test_train_split(self, X: list, y: list, ratio: float=0.8) -> list:
        """

        :param X:
        :param y:
        :param ratio:
        :return:
        """

        test_X, train_X = X[:int(len(X) * ratio)], X[int(len(X) * ratio):]
        test_y, train_y = y[:int(len(y) * ratio)], y[int(len(y) * ratio):]

        return test_X, train_X, test_y, train_y

    @property
    def accuracy(self):
        """

        :return:
        """
        pass

    @property
    def sensitivity(self):
        """

        :return:
        """
        pass


class k_nearest_neighbor(Model):
    """ K-nearest neighbors ML model"""

    def __init__(self):
        self.data = None

    def fit(self, X: list, y: list) -> None:
        """

        :param X:
        :param y:
        :return:
        """
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length")

        if not all(isinstance(point, type(X[0])) for point in X):
            raise TypeError("Every value in X has to be of the same type")

        self.data = list(zip(X, y))

    def predict(self, x: list or float, k: int = None):
        """

        :param x:
        :param k:
        :return:
        """
        if self.data is None:
            raise ValueError("model needs to be fit before predictions can be made, see model.fit(X, y)")

        if not isinstance(x, type(self.data[0][0])):
            raise TypeError("variable x is not of the same type as variables in the data set")

        if k is None:
            k = len(set([label[1] for label in self.data])) + 1

        try:
            neighbors = sorted(self.data, key=lambda a: distances.euclidian(a[0], x))[:k]
        except TypeError:
            print("Distance calculations are not supported for type {}".format(type(x)))  # TODO Switch to logging module instead of print
            raise

        counted_labels = Counter([label[1] for label in neighbors])
        return max(counted_labels.items(), key=operator.itemgetter(1))[0]


MODEL = k_nearest_neighbor()
MODEL.fit([1.2, 1.1, 1.5, 1.6], ['a', 'a', 'b', 'b'])
print(MODEL.predict(1.3))
print(MODEL.test_train_split([1.2, 1.1, 1.5, 1.6], ['a', 'a', 'b', 'b']))
# >> b'
