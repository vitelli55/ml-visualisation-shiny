import pandas as pd
import numpy as np

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x_values
        y = points.iloc[i].y_values

        error = y - (m_now * x + b_now)
        m_gradient += -(2/n) * x * error
        b_gradient += -(2/n) * error

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

def kmeans_clustering(k, X, max_iterations=200):
    centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(k, X.shape[1])) #ensures centroids are contained within data range
    for _ in range(max_iterations):
        y = []

        for data_point in X:
            distances = np.sqrt(np.sum((centroids - data_point)**2, axis=1))                #KMeansClustering.euclidean_distance(data_point, self.centroids)
            cluster_num = np.argmin(distances)
            y.append(cluster_num)

        y= np.array(y)

        cluster_indices = []

        for i in range(k):
            cluster_indices.append(np.argwhere(y==i))

        cluster_centers = []

        for i, indices in enumerate(cluster_indices):
            if len(indices)==0:
                cluster_centers.append(centroids[i])
            else:
                cluster_centers.append(np.mean(X[indices], axis=0)[0])

        if np.max(centroids - np.array(cluster_centers)) < 0.0001:
            break
        else:
            centroids = np.array(cluster_centers)
    return y