from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
import numpy as np
import tensorflow as tf

vector_arrays = [
    np.array(
        [[-1, 1, 1],
         [-1, 1, 1],
         [-1, 1, 1],
         [1, 2, 0],
         [1, 0, 2],
         [5, -2, 5],
         [6, 2, 4],
         [3, 0, 3],
         [3, -3, -2],
         [4, 2, -2],
         [0, 2, 0],
         [0, -5, 0],
         [0, -2, 0]], dtype=np.float),
    np.array(
        [[-2, 1, 1],
         [0, 3, 0],
         [0, -5, 0],
         [0, 0, 0]], dtype=np.float)]

label_arrays = [np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]),
                np.array([0, 1, 1, 1])]

grft = GroupRepelFeatureTransformer(
    3, 2, nonlinear_layer_count=1, iteration_count=50, reinitialize_on_fit=False)

import matplotlib.pyplot as plt

plt.ion()

for i in range(100):
    grft.fit(vector_arrays, label_arrays)
    result = grft.transform(vector_arrays[0:1])[0]
    plt.clf()
    a = result[:, 0]
    plt.scatter(result[:, 0], result[:, 1], c=label_arrays[0])
    plt.pause(0.05)
