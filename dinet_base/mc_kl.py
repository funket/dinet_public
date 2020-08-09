import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .settings import PRECISION


def calculate_sampled_kl_of_generalized_gaussians(sigma_f, sigma_g, mu_f, mu_g, beta=8, number_of_samples=1000):
    multivariate_normal = tfp.distributions.MultivariateNormalDiag(loc=mu_f, scale_diag=sigma_f)

    alpha_g = tf.constant(np.sqrt(2), dtype=PRECISION) * sigma_g
    alpha_f = tf.constant(np.sqrt(2), dtype=PRECISION) * sigma_f

    random_values = tf.cast(multivariate_normal.sample(sample_shape=number_of_samples), dtype=PRECISION)

    r_f = (random_values - mu_f) / alpha_f
    r_g = (random_values - mu_g) / alpha_g

    sampling_beta = tf.constant(2, dtype=PRECISION)
    beta = tf.constant(beta, dtype=PRECISION)

    sampling_correction = beta / sampling_beta * tf.math.exp(tf.math.lgamma(1 / sampling_beta)) / tf.math.exp(
        tf.math.lgamma(1 / beta)) * tf.math.exp(tf.reduce_sum(tf.pow(r_f, sampling_beta) - tf.pow(r_f, beta), axis=-1))

    return -tf.reduce_mean(sampling_correction * tf.reduce_sum(
        (tf.math.log(alpha_f) - tf.math.log(alpha_g) + (tf.pow(r_f, beta) - tf.pow(r_g, beta))),
        axis=-1), axis=0)


class TwoDModelMCMC:
    def __init__(self, number_of_nodes, alpha=None, mus=None, sigmas=None, beta=4):
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
            self.alpha = tf.Variable(2.5, dtype=PRECISION)  # average shortest path
        else:
            self.alpha = tf.Variable(alpha, dtype=PRECISION)

        self.beta = beta

    def __call__(self, edges):
        mu_alpha = tf.gather(self.mus, edges[0])
        mu_beta = tf.gather(self.mus, edges[1])
        sigma_alpha = tf.gather(self.sigmas, edges[0])
        sigma_beta = tf.gather(self.sigmas, edges[1])
        return self.alpha * calculate_sampled_kl_of_generalized_gaussians(sigma_alpha, sigma_beta, mu_alpha, mu_beta,
                                                                          self.beta)

    def standard_call(self, edges):
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
