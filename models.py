import pandas as pd
import numpy as np

class LinearRegression():

    def __init__(self):
        self.m = 0
        self.b = 0
        self.data = None
        self.L = 0.01 #learning rate

    def gradient_desc(self, m_now, b_now):

        m_gradient = 0
        b_gradient = 0

        n = len(self.data)

        for i in range(n):
            self.x = self.data.iloc[i,0]
            self.y = self.data.iloc[i,1]

            error = self.y - (m_now * self.x + b_now)
            m_gradient += -(2/n) * self.x * error
            b_gradient += -(2/n) * error
        
        self.m = m_now - m_gradient * self.L
        self.b = b_now - b_gradient * self.L
        return self.m, self.b
    
    def fit(self, data, epochs=300):
        self.data = data

        self.m_values = []
        self.b_values = []

        for i in range(epochs):
            self.m, self.b = self.gradient_desc(self.m, self.b)
            if i %50 ==0:
                self.m_values.append(self.m)
                self.b_values.append(self.b)

class KMeansClustering():

    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        self.history = []


    def fit(self, X, max_iterations=200):
        self.history = []

        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1])) #ensures centroids are contained within data range
        for _ in range(max_iterations):
            y = []

            for data_point in X:
                distances = np.sqrt(np.sum((self.centroids - data_point)**2, axis=1))  
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y= np.array(y)

            self.history.append((self.centroids.copy(), y.copy()))

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y==i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices)==0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 1e-6:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y