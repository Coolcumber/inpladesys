import numpy as np
import tensorflow as tf
from inpladesys.models.feature_transformation.abstract_feature_transformer import \
    AbstractFeatureTransformer
from typing import List
from inpladesys.datatypes import Dataset


class GroupRepelFeatureTransformer(AbstractFeatureTransformer):
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 nonlinear_layer_count=3,
                 iteration_count=10000):
        self.iteration_count = iteration_count

        self.x, self.labels, self.y = [None] * 3
        self.train_step_nd, self.loss_nd = [None] * 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self._build_graph(input_dimension, output_dimension,
                              nonlinear_layer_count)

    def fit(self, X: List[np.ndarray], G: List[np.ndarray]):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            ds = Dataset(X[:], G[:])
            for i in range(self.iteration_count):
                ds.shuffle()
                for x, labels in ds:
                    fetches = [self.train_step_nd, self.loss_nd, self.y]
                    feed_dict = {self.x: x, self.labels: labels}
                    _, cost, y = self.sess.run(fetches, feed_dict)
                    if i % 100 == 9:
                        print(cost)
                        print(y)

    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        with self.graph.as_default():
            if X is not list:
                return self.sess.run([self.y], {self.x: X})
            return [self.sess.run([self.y], {self.x: x}) for x in X]

    def _build_graph(self, x_dim, y_dim, nonlinear_layer_count):
        def fc(in_size: int, out_size: int):
            return tf.Variable(
                tf.truncated_normal([in_size, out_size], 0, 0.1))

        def bias(size: int):
            return tf.Variable(tf.truncated_normal([size], 0, 0.1))

        def transform(x):
            nlc = nonlinear_layer_count
            w = [fc(x_dim, 2 * x_dim)]
            w += [fc(2 * x_dim, 2 * x_dim) for i in range(nlc - 1)]
            b = [bias(2 * x_dim)]
            b += [bias(2 * x_dim) for i in range(nlc - 1)]
            for i in range(len(w)):
                x = tf.nn.tanh(tf.matmul(x, w[i]) + b[i])
            prevdim = 2 * x_dim if nlc > 0 else x_dim
            x = tf.matmul(x, fc(prevdim, y_dim))
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

        def get_inner_loss(y, labels, centroids):
            def loss(i):
                group = extract_group(y, labels, i)
                return tf.reduce_mean((group - centroids[i])**2)

            return tf.while_loop(
                lambda i, e: tf.less(i, centroid_count),
                lambda i, e: (i + 1, e + loss(i)),
                [tf.constant(0), tf.constant(.0)])[1]

        def get_outer_loss(centroids):
            def get_energy(i):  # TODO: optimize - halve the calculations
                others = tf.concat(
                    [centroids[:i, :], centroids[i + 1:, :]], axis=0)
                diffs = others - centroids[i, :]
                distances = tf.reduce_sum(diffs**2, axis=1)
                return tf.reduce_mean(1/distances)
                #distances = tf.sqrt(tf.reduce_sum(diffs**2, axis=1))
                #return tf.reduce_mean(1 / distances)

            return tf.while_loop(
                lambda i, e: tf.less(i, centroid_count),
                lambda i, e: (i + 1, e + get_energy(i)),
                [tf.constant(0), tf.constant(.0)], )[1]

        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, x_dim])
        labels = tf.placeholder(tf.int32, shape=[None])

        # Output
        y = transform(x)

        # Training
        centroid_count = tf.reduce_max(labels) + 1
        centroids = get_centroids(y, labels)

        inner_loss = get_inner_loss(y, labels, centroids)
        outer_loss = get_outer_loss(centroids)

        loss = inner_loss + outer_loss
        optimizer = tf.train.RMSPropOptimizer(3e-4, decay=0.9)
        train_step = optimizer.minimize(loss)

        # Variables for evaluation
        self.x, self.labels = x, labels
        self.y = y
        self.train_step_nd, self.loss_nd = train_step, loss
        self.centroids = centroids


"""if __name__ == "__main__":
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
         [0, -2, 0]],
        dtype=np.float)
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])

    cpn = GroupRepelFeatureTransformer(3, 2, iteration_count=100)

    result = cpn.fit([vectors], [labels])

    print(result[:, 0], result[:, 1])

    import matplotlib.pyplot as plt
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()"""
