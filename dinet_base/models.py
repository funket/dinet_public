import tensorflow as tf

from .settings import PRECISION


class Model:
    def __init__(self, number_of_nodes, alpha=None, mus=None, sigmas=None):
        self.N = tf.constant(number_of_nodes)

        shape = [number_of_nodes]
        random_init = tf.random_uniform_initializer()

        if mus is None:
            random_init.minval = 0
            random_init.maxval = 10
            self.mus = tf.Variable(random_init(shape=shape, dtype=PRECISION))
        else:
            self.mus = tf.Variable(mus, dtype=PRECISION)

        if sigmas is None:
            random_init.minval = 4
            random_init.maxval = 7
            self.sigmas = tf.Variable(random_init(shape=shape, dtype=PRECISION))
        else:
            self.sigmas = tf.Variable(sigmas, dtype=PRECISION)

        if alpha is None:
            self.alpha = tf.Variable(2.5, dtype=PRECISION)
        else:
            self.alpha = tf.Variable(alpha, dtype=PRECISION)

    def __call__(self, edges):
        mu_alpha = tf.gather(self.mus, edges[0])
        mu_beta = tf.gather(self.mus, edges[1])
        sigma_alpha = tf.gather(self.sigmas, edges[0])
        sigma_beta = tf.gather(self.sigmas, edges[1])

        return self.alpha * (
                tf.square(mu_alpha - mu_beta) / (2 * tf.square(sigma_beta))
                + .5 * (
                        tf.square(sigma_alpha) / tf.square(sigma_beta)
                        - 1
                        - tf.math.log(tf.square(sigma_alpha) / tf.square(sigma_beta))
                ))

    def kl(self, edge):
        return self.alpha * (tf.square(self.mus[edge[0]] - self.mus[edge[1]]) / (
                2 * tf.square(self.sigmas[edge[1]])) + .5 * (
                                     tf.square(self.sigmas[edge[0]]) / tf.square(self.sigmas[edge[1]])
                                     - 1
                                     - tf.math.log(tf.square(self.sigmas[edge[0]]) / tf.square(self.sigmas[edge[1]]))
                             ))

    @property
    def state(self):
        return self.alpha.numpy(), self.mus.numpy(), self.sigmas.numpy()

    def set_state(self, alpha, mus, sigmas):
        self.alpha = tf.Variable(alpha, dtype=PRECISION)
        self.mus = tf.Variable(mus, dtype=PRECISION)
        self.sigmas = tf.Variable(sigmas, dtype=PRECISION)


class TwoDModel:
    def __init__(self, number_of_nodes, alpha=None, mus=None, sigmas=None):
        self.N = tf.constant(number_of_nodes)
        shape = [number_of_nodes, 2]
        random_init = tf.random_uniform_initializer()

        if mus is None:
            random_init.minval = 0
            random_init.maxval = 10
            self.mus = tf.Variable(random_init(shape=shape, dtype=PRECISION))
        else:
            self.mus = tf.Variable(mus, dtype=PRECISION)

        if sigmas is None:
            random_init.minval = 4
            random_init.maxval = 7
            self.sigmas = tf.Variable(random_init(shape=shape, dtype=PRECISION))
        else:
            self.sigmas = tf.Variable(sigmas, dtype=PRECISION)

        if alpha is None:
            self.alpha = tf.Variable(2.5, dtype=PRECISION)
        else:
            self.alpha = tf.Variable(alpha, dtype=PRECISION)

    def __call__(self, edges):
        mu_alpha = tf.gather(self.mus, edges[0])
        mu_beta = tf.gather(self.mus, edges[1])
        sigma_alpha = tf.gather(self.sigmas, edges[0])
        sigma_beta = tf.gather(self.sigmas, edges[1])
        return self.alpha * .5 * (
                tf.reduce_sum(tf.square(sigma_alpha / sigma_beta)
                              + tf.square(mu_alpha - mu_beta) / tf.square(sigma_beta),
                              axis=1)
                - 2  # k=2
                + tf.math.log(tf.square(tf.reduce_prod(sigma_beta, axis=1) / tf.reduce_prod(sigma_alpha, axis=1)))
        )

    def kl(self, edge):
        return self.alpha * .5 * (
                tf.square(self.sigmas[edge[0], 0] / self.sigmas[edge[1], 0])
                + tf.square(self.sigmas[edge[0], 1] / self.sigmas[edge[1], 1])
                + tf.square((self.mus[edge[1], 0] - self.mus[edge[0], 0]) / self.sigmas[edge[1], 0])
                + tf.square((self.mus[edge[1], 1] - self.mus[edge[0], 1]) / self.sigmas[edge[1], 1])
                - 2  # k=2
                + tf.math.log(tf.square(
            self.sigmas[edge[1], 0] * self.sigmas[edge[1], 1] / (
                    self.sigmas[edge[0], 0] * self.sigmas[edge[0], 1])))
        )

    @property
    def state(self):
        return self.alpha.numpy(), self.mus.numpy(), self.sigmas.numpy()

    def set_state(self, alpha, mus, sigmas):
        self.alpha = tf.Variable(alpha, dtype=PRECISION)
        self.mus = tf.Variable(mus, dtype=PRECISION)
        self.sigmas = tf.Variable(sigmas, dtype=PRECISION)


class GeneralDModel:
    def __init__(self, number_of_nodes, embedding_dimension, alpha=None, mus=None, sigmas=None):
        self.N = tf.constant(number_of_nodes)
        self.k = tf.constant(embedding_dimension, dtype=PRECISION)
        shape = [number_of_nodes, embedding_dimension]
        random_init = tf.random_uniform_initializer()

        if mus is None:
            random_init.minval = 0
            random_init.maxval = 10
            self.mus = tf.Variable(random_init(shape=shape, dtype=PRECISION))
        else:
            self.mus = tf.Variable(mus, dtype=PRECISION)

        if sigmas is None:
            random_init.minval = 4
            random_init.maxval = 7
            self.sigmas = tf.Variable(random_init(shape=shape, dtype=PRECISION))
        else:
            self.sigmas = tf.Variable(sigmas, dtype=PRECISION)

        if alpha is None:
            self.alpha = tf.Variable(2.5, dtype=PRECISION)  # average shortest path
        else:
            self.alpha = tf.Variable(alpha, dtype=PRECISION)

    def __call__(self, edges):
        mu_alpha = tf.gather(self.mus, edges[0])
        mu_beta = tf.gather(self.mus, edges[1])
        sigma_alpha = tf.gather(self.sigmas, edges[0])
        sigma_beta = tf.gather(self.sigmas, edges[1])
        return self.alpha * .5 * (
                tf.reduce_sum(tf.square(sigma_alpha / sigma_beta)
                              + tf.square(mu_alpha - mu_beta) / tf.square(sigma_beta),
                              axis=1)
                - self.k
                + tf.math.log(tf.square(tf.reduce_prod(sigma_beta, axis=1) / tf.reduce_prod(sigma_alpha, axis=1)))
        )

    @property
    def state(self):
        return self.alpha.numpy(), self.mus.numpy(), self.sigmas.numpy()

    def set_state(self, alpha, mus, sigmas):
        self.alpha = tf.Variable(alpha, dtype=PRECISION)
        self.mus = tf.Variable(mus, dtype=PRECISION)
        self.sigmas = tf.Variable(sigmas, dtype=PRECISION)
