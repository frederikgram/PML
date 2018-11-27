class Model:
    """Base class for ML models, this contains various functions with wide use cases"""

    @staticmethod
    def test_train_split(X: list, y: list, ratio: float=0.8) -> list:
        """

        :param X: List of data
        :param y: List of labels
        :param ratio: Split percentage as a float, 0.8 would give 80% train, 20% test from the original data set
        :return: 4 lists of data test_X, train_X, test_y, train_y split from X and y
        """

        test_X, train_X = X[:int(len(X) * ratio)], X[int(len(X) * ratio):]
        test_y, train_y = y[:int(len(y) * ratio)], y[int(len(y) * ratio):]

        return test_X, train_X, test_y, train_y

    @staticmethod
    def visualize2D(X: list, y: list, colors: list=None) -> None:
        """ Visualize 2D data using matplotlib
        :param X: List of data
        :param y: List of labels
        :param colors: List of RGB tuples to apply to data points
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Cannot import matplotlib")
            raise

        fig, ax = plt.subplots()
        ax.scatter(X, y, c=colors)

        plt.show()


class Classifier(Model):
    """ Classifier type machine learning models"""

    pass


class Cluster(Model):
    """ Clustering type machine learning models"""

    pass
