import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

def linear_regression_df():
    rand_m = np.random.uniform(-5,5)
    x_data = [np.random.randint(50, 200) for _ in range(100)]

    df = pd.DataFrame({
        "x_values": x_data,
        "y_values": [int(rand_m * xi + np.random.normal(2,15)) for xi in x_data] 
    })
    return df

def kmeans_df():
    blobs = make_blobs(n_samples=100, n_features=2, centers=3)
    random_pts = blobs[0]
    df = pd.DataFrame({"x_values": random_pts[:, 0], "y_values": random_pts[:, 1]})

    return df.round(2)



