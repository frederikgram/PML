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


class Classifier(Model):
    """"""

    def generate_random_dataset(self, n_data: int, n_labels:int, data_range: tuple=(0, 10)) -> list:
        """

        :param n_data:
        :param n_labels:
        :param data_range:
        :return:
        """
        import string
        import random

        label_ratio = data_range[1] / n_labels

        X = [random.uniform(*data_range) for _ in range(n_data)]
        y = [string.ascii_letters[int(i / label_ratio)] for i in X]

        return X, y
