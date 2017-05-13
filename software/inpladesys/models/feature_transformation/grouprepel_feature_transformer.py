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
                 iteration_count=10000,
                 regularization=1):
        self.iteration_count = iteration_count

        self.x, self.labels, self.y = [None] * 3
        self.train_step_nd, self.loss_nd = [None] * 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self._build_graph(input_dimension, output_dimension,
                              nonlinear_layer_count, regularization)

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

    def _build_graph(self, x_dim, y_dim, nonlinear_layer_count, regularization):
        def fc(in_size: int, out_size: int):
            w = tf.Variable(tf.truncated_normal([in_size, out_size], 0, 0.1))
            regpars.append(w)
            return w

        def bias(size: int):
            b = tf.Variable(tf.truncated_normal([size], 0, 0.1))
            regpars.append(b)
            return b

        regpars = []

        def transform(x):
            nlc = nonlinear_layer_count
            if nlc == 0:
                x = tf.matmul(x, fc(x_dim, y_dim))
            else:
                w = [fc(x_dim, 2 * x_dim)]
                w += [fc(2 * x_dim, 2 * x_dim) for i in range(nlc - 1)]
                b = [bias(2 * x_dim)]
                b += [bias(2 * x_dim) for i in range(nlc - 1)]
                for wi, bi in zip(w, b):
                    x = tf.nn.tanh(tf.matmul(x, wi) + bi)
                x = tf.matmul(x, fc(2 * x_dim, y_dim))
            return x

        def extract_group(y, labels, label):
            indices = tf.reshape(tf.where(tf.equal(labels, label)), [-1])
            return tf.gather(y, indices)

        centroid_count = None

        def tensor_for(imin, isup, body, body_var, shape_invar=None):
            if type(imin) == int:
                imin = tf.constant(imin)
            if type(isup) == int:
                imax = tf.constant(isup)
            if shape_invar is not None:
                shape_invar = [tf.TensorShape([]), shape_invar]
            return tf.while_loop(
                lambda i, b: tf.less(i, isup),
                lambda i, b: (i + 1, body(i, b)),
                [imin, body_var],
                shape_invar
            )[1]

        def get_centroids(y, labels):
            def centroid(i):
                return tf.reduce_mean(extract_group(y, labels, i), axis=0)

            return tensor_for(1, centroid_count,
                              lambda i, c: tf.concat(
                                  [c, tf.reshape(centroid(i), [1, y_dim])], axis=0),
                              tf.reshape(centroid(0), [1, y_dim]),
                              tf.TensorShape([None, y_dim]))

        def get_inner_loss(y, labels, centroids):
            def loss(i):
                group = extract_group(y, labels, i)
                return tf.reduce_mean((group - centroids[i])**2)
            return tensor_for(0, centroid_count,
                              lambda i, e: e + loss(i),
                              tf.constant(.0))

        def get_outer_loss(centroids):
            def get_part(i):
                diffs = centroids[i + 1:, :] - centroids[i, :]
                distances = tf.reduce_sum(diffs**2, axis=0)
                return tf.reduce_sum(1/distances)
                # return tf.reduce_sum(1 / distances)
                # distances = tf.sqrt(tf.reduce_sum(diffs**2, axis=1))
                # return tf.reduce_mean(1 / distances)
            return tensor_for(0, centroid_count - 1,
                              lambda i, e: e + get_part(i),
                              tf.constant(.0)) / tf.cast(centroid_count * (centroid_count - 1), tf.float32)

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
        if regularization > 0:
            for p in regpars:
                loss += regularization * tf.reduce_mean(p**2)

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
