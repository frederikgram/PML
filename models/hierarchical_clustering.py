from models import base_models
from tools import distances


class Hierarchical_clustering(base_models.Cluster):
    """ Base model for Hierarchical clustering ML models"""

    def __init__(self):
        """ Initializes a table for mapping linkage argument[str] to functions """

        self.linkage_function_table = {"single": self.single_linkage_distance,
                                      "average": self.average_linkage_distance,
                                     "complete": self.complete_linkage_distance}

    @staticmethod
    def single_linkage_distance(a: list, b: list) -> float:
        """ Distance between the closest points of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """
        pass

    @staticmethod
    def average_linkage_distance(a: list, b: list) -> float:
        """ Distance between the average point of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """
        pass

    @staticmethod
    def complete_linkage_distance(a: list, b: list) -> float:
        """ Distance between the two farthest points of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """
        pass


class HAC(Hierarchical_clustering):
    """ Hierarchical agglomerative clustering ML model"""

    def predict(self, X: list, n_clusters: int, linkage: str="average") -> list:
        """

        :param X: List of data
        :param n_clusters: Amount of clusters to split the data into
        :param linkage: What to calculate distance from (single = distance between the closest points of clusters,
                                                        average = distance between the average point of clusters,
                                                       complete = distance between the two farthest points of clusters)
        :return yield: List of clusters at n iteration
        """

        # set appropriate linkage function
        if linkage in self.linkage_function_table:
            _linkage_function = self.linkage_function_table[linkage]
        else:
            raise ValueError("linkage type {0} is not a supported linkage criteria".format(linkage))

        # Set each data point in X to be its own cluster
        # Clusters are formatted as tuple(X[n], int(cluster_label))
        clusters: list[tuple] = [(x, enum) for enum, x in enumerate(X)]
        print(clusters)

        # Begin agglomeration
        while True:
            new_clusters: list[tuple] = []

            min_distance = None
            for cluster in clusters:
                min_distance = min([_linkage_function(cluster, cluster_b) ffor cluster_b in clusetsrs])

            # Exit the generator if the agglomeration is done
            if new_clusters == clusters:
                raise GeneratorExit()

            yield clusters

h = HAC()
print(list(h.predict([1, 2, 3, 4], 2)))
