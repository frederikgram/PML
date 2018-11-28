from models import base_models
from tools import distances


class HierarchicalClustering(base_models.Cluster):
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

        min_distance = None

        for point_a in a:
            for point_b in b:

                difference = point_a - point_b

                if difference < min_distance or min_distance is None:
                    min_distance = difference

        return min_distance

    @staticmethod
    def average_linkage_distance(a: list, b: list) -> float:
        """ Distance between the average point of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """

        if len(a) != len(b):
            raise ValueError("Lists have to be of equal length")

        return sum(a) - sum(b) / len(a)

    @staticmethod
    def complete_linkage_distance(a: list, b: list) -> float:
        """ Distance between the two farthest points of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """

        max_distance = None

        for point_a in a:
            for point_b in b:

                difference = point_a - point_b

                if max_distance is None or difference > max_distance:
                    max_distance = difference

        return max_distance


class HAC(HierarchicalClustering):
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

            min_difference = None
            new_cluster = None

            for cluster_a in clusters:
                for cluster_b in [cluster for cluster in clusters if cluster != cluster_a]:
                    difference = _linkage_function(cluster_a, cluster_b)
                    if min_difference is None or difference < min_difference:
                        min_difference = difference
                        new_cluster = (cluster_a, cluster_b)

            new_clusters = [cluster for cluster in clusters if cluster not in new_cluster]
            new_clusters.append([new_cluster[0][0], new_cluster[1][0], len(new_clusters)])

            # Recalculate cluster labels
            new_clusters = [(*cluster[:-1], i) for i, cluster in enumerate(new_clusters)]

            print(new_clusters)

            input()

            # Exit the generator if the clusters have stabilized
            if new_clusters == clusters:
                raise GeneratorExit()

            # Exit the generator if agglomeration has split X into n_clusters
            if len(clusters) == n_clusters:
                raise GeneratorExit()

            yield clusters

h = HAC()
print(list(h.predict([1, 2, 3, 4], 2)))
