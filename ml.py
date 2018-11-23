from collections import Counter
import operator

import distances
import base_models


#  TODO fill out docstrings

class k_nearest_neighbor(base_models.Classifier):
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

X, y = MODEL.generate_random_dataset(10, 4)
test_X, train_X, test_y, train_y = MODEL.test_train_split(X, y)

MODEL.fit(train_X, train_y)

for entry in zip(test_X, test_y):
    prediction = MODEL.predict(entry[0])
    print("Prediction: {prediction}\nlabel: {label}\n".format(prediction=prediction, label=entry[1]))
# >> b'
