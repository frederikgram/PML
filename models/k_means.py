from models import base_models, k_nearest_neighbor
import numpy as np
import random


class Kmeans(base_models.Cluster):
    """ K-means ML Model"""

    @staticmethod
    def predict(X: list, n_clusters: int, random_seed=None, **kwargs) -> list:
        """ Splits data into clusters using k-nearest neighbors
        :param X: List of data
        :param n_clusters: Amount of clusters to split the data into
        :param random_seed: Set a seed for the randomizer to allow for deterministic results

        **kwargs: Sends **kwargs to the K-NearestNeighbors Model
        :return: List of labels (Cluster label as int)
        """

        if random_seed is not None:
            try:
                random.seed = random_seed
            except TypeError:
                print("{seed} cannot be used as a random.seed()".format(seed=random_seed))
                raise

        knn = k_nearest_neighbor.KNearestNeighbor()

        # Set initial centroids
        centroids = random.sample(X, n_clusters)

        # Recalculate centroids from the mean of their clusters, break when centroid values have stabilized
        while True:
            knn.fit(centroids, range(0, n_clusters))

            X_without_centroids = [x for x in X if x not in centroids]
            _predictions: list[int] = [knn.predict(x, k=1, **kwargs) for x in X_without_centroids]

            X_with_predictions: list[tuple] = zip(X_without_centroids, _predictions)

            # Calculate new centroids
            new_centroids = []
            for enum, centroid in enumerate(centroids):

                # find the mean of cluster[enum]
                mean = np.mean([x[0] for x in X_with_predictions if x[1] == enum] + [centroid])
                new_centroids.append(mean)

            # Check if centroids have stabilized
            if new_centroids == centroids:
                break
            else:
                centroids = new_centroids

        # Predict cluster labels using stabilized centroids
        cluster_predictions: list[int] = [knn.predict(x, k=1, **kwargs) for x in X]
        return cluster_predictions


if __name__ == "__main__":
    """ Run example """

    model = Kmeans()
    X = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    predictions = model.predict(X, 3, random_seed=9999, distance_metric=lambda a, b: abs(a - b))
    print(predictions)
