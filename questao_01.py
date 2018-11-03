#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


class DataGenerator(object):
    def __init__(self):
        self.full_data = np.array(
            [
                [-7.82, -4.58, -3.97],
                [-6.68, 3.16, 2.71],
                [4.36, -2.19, 2.09],
                [6.72, 0.88, 2.80],
                [-8.64, 3.06, 3.50],
                [-6.87, 0.57, -5.45],
                [4.47, -2.62, 5.76],
                [6.73, -2.01, 4.18],
                [-7.71, 2.34, -6.33],
                [-6.91, -0.49, -5.68],
                [6.18, 2.81, 5.82],
                [6.72, -0.93, -4.04],
                [-6.25, -0.26, 0.56],
                [-6.94, -1.22, 1.13],
                [8.09, 0.20, 2.25],
                [6.81, 0.17, -4.15],
                [-5.19, 4.24, 4.04],
                [-6.38, -1.74, 1.43],
                [4.08, 1.30, 5.33],
                [6.27, 0.93, -2.78],
            ]
        )


def main():
    generator = DataGenerator()
    data = generator.full_data
    fig = plt.figure()
    centroids_1 = np.array([[0, 0, 0], [1, 1, 1], [-1, 0, 2]])
    centroids_2 = np.array([[-0.1, 0, 0.1], [0, -0.1, 0.1], [-0.1, -0.1, 0.1]])
    kmeans_1 = KMeans(n_clusters=3, init=centroids_1)
    kmeans_2 = KMeans(n_clusters=3, init=centroids_2)
    y_pred_1 = kmeans_1.fit_predict(data)
    y_pred_2 = kmeans_2.fit_predict(data)
    ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax_1.set_title(f'3 Clusters ({kmeans_1.n_iter_} iterações)')
    ax_2.set_title(f'3 Clusters ({kmeans_2.n_iter_} iterações)')
    ax_1.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=y_pred_1, depthshade=False)
    ax_1.scatter3D(centroids_1[:, 0], centroids_1[:, 1], centroids_1[:, 2], c='r')
    ax_1.scatter3D(kmeans_1.cluster_centers_[:, 0], kmeans_1.cluster_centers_[:, 1], kmeans_1.cluster_centers_[:, 2], c='b')
    ax_2.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=y_pred_2, depthshade=False)
    ax_2.scatter3D(centroids_2[:, 0], centroids_2[:, 1], centroids_2[:, 2], c='r')
    ax_2.scatter3D(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1], kmeans_2.cluster_centers_[:, 2], c='b')
    plt.show()


if __name__ == "__main__":
    main()
