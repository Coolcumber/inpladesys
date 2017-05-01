import numpy as np
import tensorflow as tf
from inpladesys.models.feature_extraction.abstract_feature_transformer import \
    AbstractFeatureTransformer
from typing import List


class GroupRepelFeatureTransformer(AbstractFeatureTransformer):
    def __init__(self, input_dimension, output_dimension,
                 additional_layer_count=2, iteration_count=10000):
        self.iteration_count = iteration_count

        self.x, self.labels, self.y = [None] * 3
        self.train_step_nd, self.error_nd = [None] * 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self._build_graph(input_dimension, output_dimension,
                              additional_layer_count)

    def fit(self, X: List[np.ndarray], G: List[np.ndarray]):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i in range(self.iteration_count):
                for x, labels in zip(X, G):
                    feed_dict = {self.x: x, self.labels: labels}
                    _, cost, y = self.sess.run(
                        [self.train_step_nd, self.error_nd, self.y],
                        feed_dict)
                    if i % 100 == 9:
                        print(cost)
                        print(y)

    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        if X is not list:
            return self.sess.run([self.y], {self.x: X})
        with self.graph.as_default():
            return [self.sess.run([self.y], {self.x: x}) for x in X]

    def _build_graph(self, x_dim, y_dim, additional_layer_count):
        def transform(x):
            def fc(in_size: int, out_size: int):
                return tf.Variable(
                    tf.truncated_normal([in_size, out_size], 0, 0.1))

            def bias(size: int):
                return tf.Variable(tf.truncated_normal([size], 0, 0.1))

            w = [fc(x_dim, 2 * x_dim)]
            w += [fc(2 * x_dim, 2 * x_dim)
                  for i in range(additional_layer_count)]
            b = [bias(2 * x_dim)]
            b += [bias(2 * x_dim) for i in range(additional_layer_count)]
            for i in range(len(w)):
                x = tf.nn.tanh(tf.matmul(x, w[i]) + b[i])
            x = tf.matmul(x, fc(2 * x_dim, y_dim))
            return x

        def extract_group(y, labels, label):
            indices = tf.reshape(tf.where(tf.equal(labels, label)), [-1])
            return tf.gather(y, indices)

        centroid_count = None

        def get_centroids(y, labels):
            def centroid(i):
                return tf.reduce_mean(extract_group(y, labels, i), axis=0)

            return tf.while_loop(
                lambda i, c: tf.less(i, centroid_count),
                lambda i, c: (i + 1, tf.concat(
                    [c, tf.reshape(centroid(i), [1, y_dim])], axis=0)),
                [tf.constant(1), tf.reshape(centroid(0), [1, y_dim])],
                shape_invariants=[tf.TensorShape(
                    []), tf.TensorShape([None, y_dim])]
            )[1]

        def get_inner_error(y, labels, centroids):
            def loss(i):
                group = extract_group(y, labels, i)
                return tf.reduce_mean((group - centroids[i]) ** 2)

            return tf.while_loop(
                lambda i, e: tf.less(i, centroid_count),
                lambda i, e: (i + 1, e + loss(i)),
                [tf.constant(0), tf.constant(.0)],
            )[1]

        def get_outer_error(centroids):
            def get_energy(i):  # TODO: optimize - halve the calculations
                others = tf.concat(
                    [centroids[:i, :], centroids[i + 1:, :]], axis=0)
                diffs = others - centroids[i, :]
                distances = tf.sqrt(tf.reduce_sum(diffs ** 2, axis=1))
                return tf.reduce_mean(1 / distances)

            return tf.while_loop(
                lambda i, e: tf.less(i, centroid_count),
                lambda i, e: (i + 1, e + get_energy(i)),
                [tf.constant(0), tf.constant(.0)],
            )[1]

        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, x_dim])
        labels = tf.placeholder(tf.int32, shape=[None])

        # Output
        y = transform(x)

        # Training
        centroid_count = tf.reduce_max(labels) + 1
        centroids = get_centroids(y, labels)

        inner_error = get_inner_error(y, labels, centroids)
        outer_error = get_outer_error(centroids)

        error = inner_error + outer_error
        optimizer = tf.train.RMSPropOptimizer(5e-5, decay=0.9)
        train_step = optimizer.minimize(error)

        # Variables for evaluation
        self.x, self.labels = x, labels
        self.y = y
        self.train_step_nd, self.error_nd = train_step, error
        self.centroids = centroids


if __name__ == "__main__":
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

    cpn = GroupRepelFeatureTransformer(3, 2)

    cpn.fit([vectors], [labels])
