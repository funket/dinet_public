import tensorflow as tf
import numpy as np
from .evaluation import calculate_correlation
import logging

from .settings import PRECISION

logger = logging.getLogger("dinet")


def loss(kl_estimations, target_distance_to_minus_beta):
    return tf.reduce_sum(tf.square(tf.math.reciprocal(1 + kl_estimations) - target_distance_to_minus_beta))


class Training:
    def __init__(self,
                 number_of_nodes,
                 model_class,
                 model_args,
                 optimizer_string,
                 learning_rate,
                 batched_input,
                 batched_output,
                 landmarks,
                 batched_evaluation_output,
                 batched_evaluation_input=None,
                 ):

        self.number_of_nodes = number_of_nodes

        self.model = model_class(*model_args)

        if optimizer_string == "Adam":
            if len(model_args) == 1:
                self.optimizer = tf.optimizers.Adam(learning_rate)
            else:
                # add clipping for larger dimensions
                self.optimizer = tf.optimizers.Adam(learning_rate, clipvalue=.05, clipnorm=.05)
        elif optimizer_string == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate)
        elif optimizer_string == "Adagrad":
            self.optimizer = tf.optimizers.Adagrad(learning_rate)
        else:
            raise ValueError()

        self.training_epochs = 0
        self.all_losses = []

        self.training_points = []

        self.alphas = []
        self.mus = []
        self.sigmas = []

        self.trainable_variables = [
            self.model.alpha,
            self.model.mus,
            self.model.sigmas,
        ]
        # if one of these is not needed adjust reference in _train

        self.training_correlation = []

        # for higher model dimension this needs to be adjusted to always get non zero values
        # depends on precision as well
        self.epsilon = tf.constant(0.00001, dtype=PRECISION)

        self.batched_input = batched_input
        self.batched_output = batched_output
        self.batched_evaluation_input = batched_evaluation_input
        self.batched_evaluation_output = batched_evaluation_output
        self.current_batch = 0
        self.number_of_batches = len(batched_input)
        if len(batched_output) != self.number_of_batches:
            raise ValueError("Different number of batches in input and output")

        self._inputs = self.batched_input[self.current_batch]
        self._outputs = self.batched_output[self.current_batch]

        self.landmarks = landmarks

    def framed_loss(self):
        return loss(self.model(self._inputs), self._outputs)

    # @tf.function
    def _train(self, number_of_epochs=100, with_correction=False):

        for i in tf.range(number_of_epochs):

            for batch in tf.range(self.number_of_batches):
                self.current_batch = batch
                self._inputs = self.batched_input[self.current_batch]
                self._outputs = self.batched_output[self.current_batch]

                grads_and_vars = self.optimizer._compute_gradients(self.framed_loss,
                                                                   self.trainable_variables)

                # do stuff
                modified_grads = []
                # for variable_counter, grad in enumerate(grads):
                for variable_counter, (grad, _) in enumerate(grads_and_vars):
                    if with_correction:
                        if isinstance(grad, tf.IndexedSlices):

                            grad = tf.math.unsorted_segment_sum(grad.values, grad.indices, self.number_of_nodes)

                            if variable_counter == 2:  # id of sigmas in trainable variables
                                grad *= tf.square(self.model.sigmas) / 2
                            elif variable_counter == 1:  # id of mus in trainable variables
                                grad *= tf.square(self.model.sigmas)
                            else:
                                raise ValueError("Missing correction of variable references")
                        else:
                            grad = grad

                    modified_grads.append((grad, self.trainable_variables[variable_counter]))

                self.optimizer.apply_gradients(modified_grads)

                # ensure non-negative sigmas
                # sigma always need to be > 0 to avoid nan values
                self.model.sigmas.assign(tf.clip_by_value(self.model.sigmas,
                                                          self.epsilon,
                                                          tf.constant(np.inf, dtype=PRECISION)))

    def _train_non_stochastic(self, number_of_epochs=100, with_correction=False):

        for i in tf.range(number_of_epochs):

            total_modified_grads = []

            for batch in tf.range(self.number_of_batches):
                self.current_batch = batch
                self._inputs = self.batched_input[self.current_batch]
                self._outputs = self.batched_output[self.current_batch]

                grads_and_vars = self.optimizer._compute_gradients(self.framed_loss,
                                                                   self.trainable_variables)

                # do stuff
                modified_grads = []
                # for variable_counter, grad in enumerate(grads):
                for variable_counter, (grad, _) in enumerate(grads_and_vars):
                    if with_correction:
                        if isinstance(grad, tf.IndexedSlices):

                            grad = tf.math.unsorted_segment_sum(grad.values, grad.indices, self.number_of_nodes)

                            if variable_counter == 2:  # id of sigmas in trainable variables
                                grad *= tf.square(self.model.sigmas) / 2
                            elif variable_counter == 1:  # id of mus in trainable variables
                                grad *= tf.square(self.model.sigmas)
                            else:
                                raise ValueError("Missing correction of variable references")

                        else:
                            grad = grad

                    modified_grads.append(grad)

                if not total_modified_grads:
                    total_modified_grads = modified_grads
                else:
                    for variable_counter in range(len(total_modified_grads)):
                        total_modified_grads[i] += modified_grads[i]

            self.optimizer.apply_gradients(zip(total_modified_grads, self.trainable_variables))

            # ensure non-negative sigmas
            # sigma always need to be > 0 to avoid nan values
            self.model.sigmas.assign(tf.clip_by_value(self.model.sigmas,
                                                      tf.constant(0.00001, dtype=PRECISION),
                                                      tf.constant(np.inf, dtype=PRECISION)))

    def train(self, number_of_loops=10, number_of_epochs=1, with_correction=False, stochastic=True):
        for i in range(number_of_loops):
            self.training_epochs += number_of_epochs

            logger.debug("Training loop " + str(i))
            if stochastic:
                self._train(number_of_epochs, with_correction)
            else:
                self._train_non_stochastic(number_of_epochs, with_correction)

            self.all_losses.append(self.framed_loss())

            self.training_points.append(self.training_epochs)
            # self.model_states.append(self.model.state)
            alpha, mu, sigma = self.model.state
            self.alphas.append(alpha)
            self.mus.append(mu)
            self.sigmas.append(sigma)

            self.training_correlation.append(
                calculate_correlation(self.model,
                                      self.batched_input,
                                      self.batched_output,
                                      self.batched_evaluation_output,
                                      self.batched_evaluation_input,
                                      ))

            logger.debug("Loss:" + str(self.all_losses[-1].numpy())
                         + "\t Training correlation " + str(self.training_correlation[-1]))

    def save(self, file):

        np.savez_compressed(file,
                            all_losses=np.array(self.all_losses),
                            alphas=np.array(self.alphas),
                            mus=np.array(self.mus),
                            sigmas=np.array(self.sigmas),
                            training_correlation=np.array(self.training_correlation),
                            )

    def save_model_with_best_loss(self, file):

        position_of_best_loss = np.argmin(self.all_losses)

        np.savez_compressed(file,
                            best_loss=np.array([self.all_losses[position_of_best_loss]]),
                            alphas=np.array([self.alphas[position_of_best_loss]]),
                            mus=np.array([self.mus[position_of_best_loss]]),
                            sigmas=np.array([self.sigmas[position_of_best_loss]]),
                            training_correlation=np.array([self.training_correlation[position_of_best_loss]]),
                            )

    def load(self, file):

        variables = np.load(file)
        self.all_losses = variables["all_losses"]
        self.alphas = variables["alphas"]
        self.mus = variables["mus"]
        self.sigmas = variables["sigmas"]
        self.training_correlation = variables["training_correlation"]


class TrainingDataset:
    def __init__(self,
                 number_of_nodes,
                 model_class,
                 model_args,
                 optimizer_string,
                 learning_rate,
                 dataset,
                 batched_input,
                 batched_output,
                 raw_batched_output,
                 landmarks):

        self.number_of_nodes = number_of_nodes

        self.model = model_class(*model_args)

        if optimizer_string == "Adam":
            if len(model_args) == 1:
                self.optimizer = tf.optimizers.Adam(learning_rate)
            else:
                # add clipping for larger dimensions
                self.optimizer = tf.optimizers.Adam(learning_rate, clipvalue=.05)
        elif optimizer_string == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate)
        elif optimizer_string == "Adagrad":
            self.optimizer = tf.optimizers.Adagrad(learning_rate)
        else:
            raise ValueError()

        self.training_epochs = 0
        self.all_losses = []

        self.training_points = []

        self.alphas = []
        self.mus = []
        self.sigmas = []

        self.trainable_variables = [
            self.model.alpha,
            self.model.mus,
            self.model.sigmas,
        ]
        # if one of these is not needed adjust reference in _train

        self.training_correlation = []

        # for higher model dimension this needs to be adjusted to always get non zero values
        # depends on precision as well
        self.epsilon = tf.constant(0.00001, dtype=PRECISION)

        self.dataset = dataset
        self.batched_input = batched_input
        self.batched_output = batched_output
        self.raw_batched_output = raw_batched_output

        self.landmarks = landmarks

    # @tf.function
    def framed_loss(self):
        return loss(self.model(self._inputs), self._outputs)

    def _train(self, number_of_epochs=100, with_correction=False):

        for i in tf.range(number_of_epochs):
            logger.debug("   Iteration completed")
            for inputs, outputs in self.dataset:
                self._inputs = tf.transpose(inputs)
                self._outputs = outputs

                grads_and_vars = self.optimizer._compute_gradients(self.framed_loss,
                                                                   self.trainable_variables)

                # do stuff
                modified_grads = []
                for variable_counter, (grad, _) in enumerate(grads_and_vars):
                    if with_correction:
                        if isinstance(grad, tf.IndexedSlices):

                            # use this to make non stochastic gradient!
                            grad = tf.math.unsorted_segment_sum(grad.values, grad.indices, self.number_of_nodes)

                            if variable_counter == 2:  # id of sigmas in trainable variables
                                grad *= tf.square(self.model.sigmas) / 2
                            elif variable_counter == 1:  # id of mus in trainable variables
                                grad *= tf.square(self.model.sigmas)
                            else:
                                raise ValueError("Missing correction of variable references")
                        else:
                            grad = grad

                    modified_grads.append((grad, self.trainable_variables[variable_counter]))

                self.optimizer.apply_gradients(modified_grads)

                # ensure non-negative sigmas
                # sigma always need to be > 0 to avoid nan values
                self.model.sigmas.assign(tf.clip_by_value(self.model.sigmas,
                                                          self.epsilon,
                                                          tf.constant(np.inf, dtype=PRECISION)))

    def train(self, number_of_loops=3, number_of_epochs=100, with_correction=False, stochastic=True):
        for i in range(number_of_loops):
            self.training_epochs += number_of_epochs

            logger.debug("Training loop " + str(i))  # time.ctime()
            self._train(number_of_epochs, with_correction)

            self.all_losses.append(self.framed_loss())

            self.training_points.append(self.training_epochs)
            # self.model_states.append(self.model.state)
            alpha, mu, sigma = self.model.state
            self.alphas.append(alpha)
            self.mus.append(mu)
            self.sigmas.append(sigma)

            self.training_correlation.append(
                calculate_correlation(self.model,
                                      self.batched_input,
                                      self.batched_output,
                                      self.raw_batched_output,
                                      ))

            logger.debug("Loss:" + str(self.all_losses[-1].numpy())
                         + "\t Training correlation " + str(self.training_correlation[-1]))

    def save(self, file):

        np.savez_compressed(file,
                            all_losses=np.array(self.all_losses),
                            alphas=np.array(self.alphas),
                            mus=np.array(self.mus),
                            sigmas=np.array(self.sigmas),
                            training_correlation=np.array(self.training_correlation),
                            )

    def load(self, file):

        variables = np.load(file)
        self.all_losses = variables["all_losses"]
        self.alphas = variables["alphas"]
        self.mus = variables["mus"]
        self.sigmas = variables["sigmas"]
        self.training_correlation = variables["training_correlation"]
