""" Simple Python implementation of a Hierarchical agglomerative clustering model """

from typing import List
from models import base_models


class HierarchicalClustering(base_models.Cluster):
    """ Base model for Hierarchical clustering ML models"""

    @staticmethod
    def single_linkage_distance(a: List, b: List, metric) -> float:
        """ Distance between the closest points of clusters """
        min_distance = None

        for point_a in a:
            for point_b in b:
                difference = metric(point_a, point_b)
                if min_distance is None or difference < min_distance:
                    min_distance = difference
        return min_distance

    @staticmethod
    def average_linkage_distance(a: List, b: List, metric) -> float:
        """ Distance between the average point of clusters """

        return metric((sum(a) / len(a)), (sum(b) / len(b)))

    @staticmethod
    def complete_linkage_distance(a: List, b: List, metric) -> float:
        """ Distance between the two farthest points of clusters """

        max_distance = None

        for point_a in a:
            for point_b in b:
                difference = abs(point_a - point_b)
                if max_distance is None or difference > max_distance:
                    max_distance = difference

        return max_distance


class HAC(HierarchicalClustering):
    """ Hierarchical agglomerative clustering ML model"""

    def predict(self, X: List, n_clusters: int = 2) -> List:
        """ Combines values of X into n_clusters using distance computations

        :param X: List of data
        :param n_clusters: Amount of clusters to condense the data into
        :return yield: List of clusters at n iteration
        """

        # Set each data point in X to be its own cluster
        # Clusters are formatted as tuple(list(X[n]), int(cluster_label))
        clusters = [([x], enum) for enum, x in enumerate(X)]

        # Begin agglomeration
        while True:

            min_difference = None
            clusters_to_merge = None

            # Find which clusters are the best to merge based on linkage criteria
            for cluster_a in clusters:
                for cluster_b in [cluster for cluster in clusters if cluster != cluster_a]:

                    difference = self.average_linkage_function(cluster_a[0], cluster_b[0], metric=lambda a, b: abs(a - b))

                    if min_difference is None or difference < min_difference:

                        min_difference = difference
                        clusters_to_merge = (cluster_a, cluster_b)

            # Merge clusters
            new_clusters = [cluster for cluster in clusters if cluster not in clusters_to_merge]
            new_clusters.append((clusters_to_merge[0][0] + clusters_to_merge[1][0], len(new_clusters) + 1))

            # Recalculate cluster labels
            new_clusters = [(cluster[0], _enum) for _enum, cluster in enumerate(new_clusters)]

            # Exit the generator if the clusters have stabilized
            if new_clusters == clusters:
                raise StopIteration

            # Exit the generator if agglomeration has split X into n_clusters
            if len(new_clusters) + 1 == n_clusters:
                raise StopIteration

            # Fit label to original input order
            labels = list()
            for x in X:
                [labels.append(cluster[1]) for cluster in new_clusters if x in cluster[0]]

            clusters = new_clusters
            yield labels
