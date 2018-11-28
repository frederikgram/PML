from models import base_models


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

                difference = abs(point_a - point_b)

                if min_distance is None or difference < min_distance:
                    min_distance = difference

        return min_distance

    @staticmethod
    def average_linkage_distance(a: list, b: list) -> float:
        """ Distance between the average point of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """

        return abs((sum(a) / len(a)) - (sum(b) / len(b)))

    @staticmethod
    def complete_linkage_distance(a: list, b: list) -> float:
        """ Distance between the two farthest points of clusters

        :param a: List of cluster values
        :param b: List of cluster values
        """

        max_distance = None

        for point_a in a:
            for point_b in b:

                difference = abs(point_a - point_b)

                if max_distance is None or difference > max_distance:
                    max_distance = difference

        return max_distance


class HAC(HierarchicalClustering):
    """ Hierarchical agglomerative clustering ML model"""

    def predict(self, X: list, n_clusters: int=2, linkage: str="average") -> list:
        """

        :param X: List of data
        :param n_clusters: Amount of clusters to split the data into (DEFAULT VALUE = 2)
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
        # Clusters are formatted as tuple(list(X[n]), int(cluster_label))
        clusters: list[tuple] = [([x], enum) for enum, x in enumerate(X)]

        # Begin agglomeration
        while True:

            min_difference = None
            clusters_to_merge = None

            # Find which clusters are the best to merge based on linkage criteria
            for cluster_a in clusters:
                for cluster_b in [cluster for cluster in clusters if cluster != cluster_a]:

                    difference = _linkage_function(cluster_a[0], cluster_b[0])

                    if min_difference is None or difference < min_difference:

                        min_difference = difference
                        clusters_to_merge = (cluster_a, cluster_b)

            # Merge clusters
            new_clusters = [cluster for cluster in clusters if cluster not in clusters_to_merge]
            new_clusters.append((clusters_to_merge[0][0] + clusters_to_merge[1][0], len(new_clusters) + 1))

            # Recalculate cluster labels
            new_clusters = [(cluster[0], enum) for enum, cluster in enumerate(new_clusters)]

            # Exit the generator if the clusters have stabilized
            if new_clusters == clusters:
                raise StopIteration

            # Exit the generator if agglomeration has split X into n_clusters
            if len(new_clusters) + 1 == n_clusters:
                raise StopIteration

            # Fit label to original input order
            labels = []
            for x in X:
                [labels.append(cluster[1]) for cluster in new_clusters if x in cluster[0]]

            clusters = new_clusters
            yield labels


if __name__ == "__main__":
    """ Run example """
    h = HAC()
    for enum, step in enumerate(h.predict(X=[1, 2, 3, 4, 5, 6, 7, 8,8, 9, 10], n_clusters=4, linkage="single")):
        print("Step: {enum}\nData:   {data}\nLabels: {labels}\n".format(enum=enum, data=[1, 2, 3, 4, 5, 6, 7, 8,8, 9, 10], labels=step))

    print("Linkage used: single")
