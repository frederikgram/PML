""" Simple Python implementation of a K-Means clustering model"""

import numpy as np
import random

from models import base_models, k_nearest_neighbor
from typing import List


class Kmeans(base_models.Cluster):
    """ K-means ML Model"""

    @staticmethod
    def predict(X: List, n_clusters: int, seed=None, **kwargs) -> List:
        """ Splits data into clusters using k-nearest neighbors
        :param X: List of data
        :param n_clusters: Amount of clusters to split the data into
        :param seed: Set a seed for the randomizer to allow for deterministic results

        **kwargs: Sends **kwargs to the K-NearestNeighbors Model
        :return: List of labels (Cluster label as int)
        """

        # Setup Initial Variables
        random.seed = seed
        knn = k_nearest_neighbor.KNearestNeighbor()

        # Set initial centroids
        centroids = random.sample(X, n_clusters)

        # Recalculate centroids from the mean of their clusters
        # Break when centroid values have stabilized
        while True:
            knn.fit(centroids, range(0, n_clusters))

            X_without_centroids = [x for x in X if x not in centroids]
            predictions = [knn.predict(x, k=1, **kwargs) for x in X_without_centroids]

            X_with_predictions = zip(X_without_centroids, predictions)

            # Calculate new centroids
            new_centroids = list()
            for enum, centroid in enumerate(centroids):
                # find the mean of cluster[enum]
                mean = np.mean([x[0] for x in X_with_predictions if x[1] == enum] + [centroid])
                new_centroids.append(mean)

            # Break if centroids have stabilized
            if new_centroids == centroids:
                break

            centroids = new_centroids

        # Predict cluster labels using stabilized centroids
        cluster_predictions = [knn.predict(x, k=1, **kwargs) for x in X]
        return cluster_predictions
