import numpy as np
import tensorflow as tf
import matplotlib as mpl
from abc import ABC, abstractmethod
from abstract_feature_extractor import AbstractFeatureExtractor
from typing import List



class GroupRepellFeatureExtractor(AbstractFeatureExtractor):
    def __init__(self, input_dimension, output_dimension):
        self.x_dim = input_dimension
        self.y_dim = output_dimension

        self.x, self.labels, self.y, self.train_step_nd, self.error_nd = [
            None] * 5

        self.graph = tf.Graph()
        # with self.graph.as_default():
        self.sess = tf.Session()
        self._build_graph(input_dimension, output_dimension)

    def fit(self, X: List[np.ndarray], C: List[np.ndarray]):
        # with self.graph.as_default():
        self.sess.run(tf.global_variables_initializer())
        for i in range(10000):
            for x, labels in zip(X, C):
                feed_dict = {self.x: x, self.labels: labels}
                _, cost, y = self.sess.run(
                    [self.train_step_nd, self.error_nd, self.y],
                    feed_dict)
                if i % 100 == 9:
                    print(cost)
                    print(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X

    def _build_graph(self, x_dim, y_dim):

        def transformation(x):
            def fc(in_size: int, out_size: int):
                return tf.Variable(tf.truncated_normal([in_size, out_size], 0, 0.1))

            def bias(size: int):
                return tf.Variable(tf.truncated_normal([size], 0, 0.1))
            layer_count = 2
            w = [fc(x_dim, 2 * x_dim)]
            w += [fc(2 * x_dim, 2 * x_dim) for i in range(layer_count)]
            b = [bias(2 * x_dim)]
            b += [bias(2 * x_dim) for i in range(layer_count)]
            for i in range(len(w)):
                x = tf.nn.tanh(tf.matmul(x, w[i]) + b[i])
            x = tf.matmul(x, fc(2 * x_dim, y_dim))
            return x

        def extract_group(y, labels, label):
            indices = tf.reshape(tf.where(tf.equal(labels, label)), [-1])
            return tf.gather(y, indices)

        def get_centroids(y, labels, centroid_count):
            def centroid(i):
                return tf.reduce_mean(extract_group(y, labels, i), axis=0)

            return tf.while_loop(
                lambda i, c: tf.less(i, centroid_count),
                lambda i, c: (i + 1,
                              tf.concat([c, tf.reshape(centroid(i), [1, y_dim])], axis=0)),
                [tf.constant(1), tf.reshape(centroid(0), [1, y_dim])],
                shape_invariants=[tf.TensorShape(
                    []), tf.TensorShape([None, y_dim])]
            )[1]

        def get_inner_error(y, labels, centroids, centroid_count):
            def loss(i):
                group = extract_group(y, labels, i)
                return tf.reduce_mean((group - centroids[i])**2)
            return tf.while_loop(
                lambda i, err: tf.less(i, centroid_count),
                lambda i, err: (i + 1, err + loss(i)),
                [tf.constant(0), tf.constant(.0)],
            )[1]

        def get_outer_error(centroids, centroid_count):
            def get_energy(i):  # TODO: optimize - halve the calculations
                others = tf.concat(
                    [centroids[:i, :], centroids[i + 1:, :]], axis=0)
                diffs = others - centroids[i, :]
                distances = tf.sqrt(tf.reduce_sum(diffs**2, axis=1))
                return tf.reduce_mean(1 / distances)
            return tf.while_loop(
                lambda i, e: tf.less(i, centroid_count),
                lambda i, e: (i + 1, e + get_energy(i)),
                [tf.constant(0), tf.constant(.0)],
            )[1]

        # Regularized parameters
        reg_params = []

        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        labels = tf.placeholder(tf.int32, shape=[None])

        # Output
        y = transformation(x)

        # Training
        centroid_count = tf.reduce_max(labels) + 1
        centroids = get_centroids(y, labels, centroid_count)
        centroid_mean = tf.reduce_mean(centroids, axis=0)

        inner_error = get_inner_error(y, labels, centroids, centroid_count)
        outer_error = get_outer_error(centroids, centroid_count)

        error = inner_error + outer_error
        optimizer = tf.train.RMSPropOptimizer(5e-5, decay=0.9)
        train_step = optimizer.minimize(error)

        self.x, self.labels, self.y = x, labels, y
        self.train_step_nd, self.error_nd = train_step, error
        self.centroids = centroids


vectors = np.array(
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
     [0, -2, 0]], dtype=np.float)
labels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])

cpn = ClustPrepNet(3, 2)

cpn.fit([vectors], [labels])
