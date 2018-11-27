from models import base_models, k_nearest_neighbor
import numpy as np
import random


class k_means(base_models.Cluster):
    """ K-means ML Model"""

    @staticmethod
    def predict(X: list, n_clusters: int) -> list:
        """ Splits data into clusters using k-nearest neighbors
        :param X: List of data
        :param n_clusters: Amount of clusters to split the data into
        :return: List of labels (Cluster label as int)
        """

        knn = k_nearest_neighbor.k_nearest_neighbor()

        # Set initial centroids
        centroids = random.sample(X, n_clusters)
        while True:
            knn.fit(centroids, range(0, n_clusters))

            X_without_centroids = [x for x in X if x not in centroids]
            predictions = [knn.predict(x, k=1) for x in X_without_centroids]

            X_with_predictions = list(zip(X_without_centroids, predictions))

            # Calculate new centroids
            new_centroids = []
            for enum, centroid in enumerate(centroids):
                mean = float(np.mean([x[0] for x in X_with_predictions if x[1] == enum] + [centroid]))
                new_centroids.append(mean)

            if new_centroids == centroids:
                break

            centroids = new_centroids

        cluster_predictions = [knn.predict(x) for x in X]
        return cluster_predictions


if __name__ == "__main__":
    """ Run test """

    model = k_means()
    X = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    predictions = model.predict(X, 3)

    print(list(zip(X, predictions)))
