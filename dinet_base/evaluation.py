import numpy as np
import scipy.stats as stats
import tensorflow as tf
import logging

RANDOM_EDGE_MODE = "random_edges"

logger = logging.getLogger("dinet")


# evaluation method used during training
def calculate_correlation(model,
                          batched_input,
                          batched_output,
                          batched_evaluation_output,
                          batched_evaluation_input=None,
                          calculate_spearman=False,
                          ):
    training_approximations = []
    # training_output = []
    evaluation_output = []

    for j in range(len(batched_input)):
        # logger.debug("  Calculating values for sample " + str(j))
        training_approximations.append(tf.math.reciprocal(1 + model(batched_input[j])).numpy())
        # try:
        #     training_output.append(batched_output[j].numpy())
        # except AttributeError:
        #     training_output.append(batched_output[j])

    # universal access to allow dataset or
    for evaluation_batch in batched_evaluation_output:
        try:
            evaluation_output.append(evaluation_batch.numpy())
        except AttributeError:
            evaluation_output.append(evaluation_batch)

    logger.debug("  Calculating correlation")
    full_training_approximation = np.concatenate(training_approximations)
    # full_training_output = np.concatenate(training_output)
    full_evaluation_output = np.concatenate(evaluation_output)

    if batched_evaluation_input is None:
        evaluation_correlation = stats.pearsonr(full_training_approximation,
                                                full_evaluation_output)
        # if calculate_spearman:
        #     evaluation_correlation = (evaluation_correlation,
        #                               stats.spearmanr(full_training_approximation, full_evaluation_output))
    else:
        evaluation_approximations = []
        for evaluation_input in batched_evaluation_input:
            evaluation_approximations.append(tf.math.reciprocal(1 + model(evaluation_input)).numpy())

        full_evaluation_approximation = np.concatenate(evaluation_approximations)

        evaluation_correlation = stats.pearsonr(full_evaluation_approximation,
                                                full_evaluation_output)

    training_correlation = (np.NaN, np.NaN)
    # training_correlation = stats.pearsonr(full_training_approximation,
    #                                       full_training_output)
    #
    # if calculate_spearman:
    #     training_correlation = (training_correlation,
    #                             stats.spearmanr(full_training_approximation, full_training_output))

    return evaluation_correlation, training_correlation