import dinet_base as dinet
import tensorflow as tf
import os
import logging
import argparse
import configparser
import numpy as np
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execution learning")
    parser.add_argument("-c", "--configfile", default="configuration.ini",
                        help="select configuration file default configuration.ini ")
    parser.add_argument("-d", "--dataset", action="store_true", default=False)
    args = parser.parse_args()

    # logging to stdout and file
    config = configparser.ConfigParser()

    # read config to know path to store log file
    config.read(args.configfile)

    # create formatter and add it to the handlers
    # additional format options %(filename)s - %(lineno)d \t
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # create file handler which logs even debug messages
    file_handler = logging.FileHandler(filename=config["global"]["global_folder"] + "/execution.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # get TF logger
    tensorflow_logger = tf.get_logger()
    tensorflow_logger.setLevel(logging.DEBUG)
    tensorflow_logger.addHandler(file_handler)

    # create dinet handler
    logger = logging.getLogger("dinet")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # add stdout
    root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # first restrict used gpus and afterwards set memory growth
    os.environ["CUDA_VISIBLE_DEVICES"] = config["global"]["gpu_number"]

    # session set-up
    tf.config.set_soft_device_placement(True)
    # tf.config.gpu.set_per_process_memory_growth(True)
    # from https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        device_name = '/device:GPU:0'
        feed_device = '/cpu:0'
    else:
        logger.warning("No GPU available")
        device_name = '/device:CPU:0'
        feed_device = '/cpu:0'

    try:
        full_graph_file = config["global"]["file"]
    except KeyError:
        full_graph_file = None
    is_graph_directed = config.getboolean("global", "directed_graph", fallback=True)

    # infer_link_prediction_embeddings = config.getboolean("global", "link_prediction", fallback=False)

    save_file_counter = 0
    training_data_configurations = {}


    def create_evaluation_input_and_output():
        evaluation_save_file = training_folder + "/evaluation_data_all_training.npz"
        try:
            evaluation_data = np.load(evaluation_save_file)
            evaluation_inputs = evaluation_data["evaluation_inputs"]
            evaluation_outputs = evaluation_data["evaluation_outputs"]

            del evaluation_data
        except FileNotFoundError:
            graph = dinet.read_graph_from_file(file, is_graph_directed)
            evaluation_inputs, evaluation_outputs, _ = dinet.generate_input(graph=graph,
                                                                            calculate_all_pairs=True,
                                                                            batches=1,
                                                                            )
            del graph
            np.savez_compressed(evaluation_save_file,
                                batched_evaluation_input=evaluation_inputs,
                                batched_evaluation_output=batched_evaluation_output,
                                )

        # remove unnecessary outer array
        evaluation_inputs = evaluation_inputs[0]

        number_of_samples = len(evaluation_inputs[0])

        # create evaluation always as dataset to save space on the gpu
        with tf.device('/device:CPU:0'):
            number_of_evaluation_batches = 6000
            input_dataset = tf.data.Dataset.from_tensor_slices((evaluation_inputs,))
            input_dataset = input_dataset.batch(int(number_of_samples / number_of_evaluation_batches))
            input_dataset = input_dataset.apply(
                tf.data.experimental.prefetch_to_device(feed_device,
                                                        buffer_size=int(
                                                            number_of_samples / number_of_evaluation_batches)))
        return input_dataset, evaluation_outputs


    with tf.device(device_name):
        number_of_loops = int(config["global"]["number_of_loops"])
        number_of_epochs = int(config["global"]["number_of_epochs"])
        embedding_dimension = int(config["global"].get("embedding_dimension", 2))
        evaluate_with_all_data = config.getboolean("global", "evaluate_with_all_data", fallback=False)
        training_folder = config["global"]["global_folder"] + "/" + config["global"]["training_data_folder"]
        train_stochastic = config.getboolean("global", "stochastic_training", fallback=True)
        use_commute_time = config.getboolean("global", "commute_distance", fallback=False)

        evaluation_input_dataset = None
        # if evaluate_with_all_data:
        #     evaluation_input_dataset, batched_evaluation_output = create_evaluation_input_and_output()

        last_file = None
        for run in config.sections():
            if run == "global":
                continue
            run_info = config[run]
            try:
                file = run_info["file"]
            except KeyError:
                if full_graph_file is None:
                    raise KeyError("No file reference found in ini global config or run config")
                file = full_graph_file

            if file != last_file:
                graph = dinet.read_graph_from_file(file, is_graph_directed)
                number_of_nodes = len(graph)
                number_of_edges = len(graph.edges)
                del graph

            save_file = run_info["save_file"]
            all_pairs = config.getboolean(run, "all_pairs", fallback=False)
            number_of_close = int(run_info.get("number_of_close", 0))
            number_of_infinities = int(run_info.get("number_of_infinities", 0))
            number_of_batches = int(run_info.get("number_of_batches", 1))
            exponent = run_info.get("exponent", 1)

            if embedding_dimension == 1:
                model = dinet.Model
                model_args = (number_of_nodes,)
            elif embedding_dimension == 2:
                model = dinet.TwoDModel
                # model = dinet.TwoDModelMCMC
                model_args = (number_of_nodes,)
            else:
                model = dinet.GeneralDModel
                model_args = (number_of_nodes, embedding_dimension)
            learning_rate = float(run_info.get("learning_rate", .01))
            optimizer_string = run_info.get("optimizer", "Adam")

            training_mode = run_info.get("training_mode", "corrected")

            if args.dataset:
                logger.debug(" Dataset execution")

            logger.debug("Execution of file " + str(file))
            logger.debug("Number of edges " + str(number_of_edges))
            logger.debug("Is directed " + str(is_graph_directed))
            logger.debug("Using all pairs " + str(all_pairs))
            logger.debug("Number of close " + str(number_of_close))
            logger.debug("Number of infinities " + str(number_of_infinities))
            logger.debug("Number of batches " + str(number_of_batches))
            logger.debug("Learning rate " + str(learning_rate))
            logger.debug("Optimizer " + optimizer_string)
            logger.debug("Exponent " + str(exponent))
            if model == dinet.TwoDModel:
                logger.debug("Model dimension  " + str(2))
            elif model == dinet.Model:
                logger.debug("Model dimension  " + str(1))
            elif model == dinet.GeneralDModel:
                logger.debug("Model dimension  " + str(model_args[1]))

            training_data_configuration = (file,
                                           all_pairs,
                                           number_of_close,
                                           number_of_infinities,
                                           number_of_batches,
                                           )

            if training_data_configuration not in training_data_configurations:
                training_data_configurations[training_data_configuration] = len(training_data_configurations)
            save_file_counter = training_data_configurations[training_data_configuration]

            training_data_save_file = training_folder + "/training_data_" + str(save_file_counter) + ".npz"
            try:
                batched_input, batched_output, landmarks = dinet.load_training_data(training_data_save_file,
                                                                                    number_of_batches)

                logger.debug("Loaded training data from file " + training_data_save_file)

                logger.debug(" ".join(map(str, ("Loaded", len(batched_input), "input batches with sample sizes",
                                                *[len(inputs[0]) for inputs in batched_input]))))
                logger.debug(" ".join(map(str, ("Loaded", len(batched_output), "output batches with sample sizes",
                                                *[len(outputs) for outputs in batched_output]))))

            except FileNotFoundError:
                graph = dinet.read_graph_from_file(file, is_graph_directed)

                batched_input, batched_output, landmarks = dinet.generate_input(
                    graph=graph,
                    calculate_all_pairs=all_pairs,
                    number_of_close=number_of_close,
                    number_of_infinities=number_of_infinities,
                    batches=number_of_batches,
                )

                del graph
                dinet.save_training_data(training_data_save_file, batched_input, batched_output, landmarks)

                logger.debug("Saved training data to file " + training_data_save_file)

                logger.debug(" ".join(map(str, ("Generated", len(batched_input), "input batches with sample sizes",
                                                *[len(inputs[0]) for inputs in batched_input]))))
                logger.debug(" ".join(map(str, ("Generated", len(batched_output), "output batches with sample sizes",
                                                *[len(outputs) for outputs in batched_output]))))

            if sum([len(output) for output in batched_output]) == 0:
                raise ValueError("Cannot train without training data")

            # execute training
            if args.dataset:
                # execute training
                raw_batched_output = [np.reciprocal(outputs) for outputs in batched_output]
            else:
                batched_input = [tf.constant(inputs) for inputs in batched_input]
                if not evaluate_with_all_data:
                    if use_commute_time:
                        batched_evaluation_output = [outputs for outputs in batched_output]
                    else:
                        batched_evaluation_output = [np.reciprocal(outputs) for outputs in batched_output]

            exponent = float(exponent)

            if args.dataset:
                batched_output = [np.reciprocal(np.power(outputs.astype(np.float32), exponent)) for outputs in
                                  batched_output]
            else:
                if exponent != 1.0:
                    batched_output = [tf.cast(np.reciprocal(np.power(outputs, exponent)), dinet.PRECISION)
                                      for outputs in batched_output]
                else:
                    batched_output = [tf.cast(np.reciprocal(outputs), dinet.PRECISION)
                                      for outputs in batched_output]

            if args.dataset:
                flat_input = np.concatenate(
                    (np.concatenate([batched_input[batch][0] for batch in range(number_of_batches)]),
                     np.concatenate([batched_input[batch][1] for batch in range(number_of_batches)])),
                    axis=0)
                flat_input = flat_input.reshape(2, -1)
                flat_output = np.concatenate(batched_output, axis=-1)

                if dinet.PRECISION == tf.float64:
                    flat_output = flat_output.astype(np.float64)

                with tf.device('/device:CPU:0'):
                    number_of_training_batches = 410
                    dataset = tf.data.Dataset.from_tensor_slices((np.transpose(flat_input), flat_output))
                    dataset = dataset.shuffle(len(flat_input), reshuffle_each_iteration=True)
                    dataset = dataset.batch(int(len(flat_output) / number_of_training_batches))
                    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                    dataset = dataset.apply(
                        tf.data.experimental.prefetch_to_device(feed_device, buffer_size=len(flat_output)))
                    # dataset = dataset.cache()

                del flat_input
                del flat_output

                trainer = dinet.TrainingDataset(
                    number_of_nodes,
                    model,
                    model_args,
                    optimizer_string,
                    learning_rate,
                    dataset,
                    batched_input,
                    batched_output,
                    raw_batched_output,
                    landmarks)
            else:

                trainer = dinet.Training(
                    number_of_nodes=number_of_nodes,
                    model_class=model,
                    model_args=model_args,
                    optimizer_string=optimizer_string,
                    learning_rate=learning_rate,
                    batched_input=batched_input,
                    batched_output=batched_output,
                    batched_evaluation_input=evaluation_input_dataset,
                    batched_evaluation_output=batched_evaluation_output,
                    landmarks=landmarks,
                )

            if training_mode == "uncorrected":
                logger.debug("Training mode uncorrected")
                trainer.train(number_of_loops, number_of_epochs, stochastic=train_stochastic)
            elif training_mode == "corrected":
                logger.debug("Training mode corrected")
                trainer.train(number_of_loops, number_of_epochs, with_correction=True, stochastic=train_stochastic)
            elif training_mode == "mixed":
                uncorrected_runs = min(10, int(.1 * number_of_loops))
                logger.debug("Training mode mixed: " + str(uncorrected_runs) + "burn in")
                logger.debug("Training loops uncorrected: " + str(uncorrected_runs) + "(burn in)")
                trainer.train(uncorrected_runs, number_of_epochs, stochastic=train_stochastic)
                logger.debug("Training loops corrected: " + str(number_of_loops - uncorrected_runs) + "(burn in)")
                trainer.train(number_of_loops - uncorrected_runs, number_of_epochs, with_correction=True,
                              stochastic=train_stochastic)
            else:
                raise ValueError("Not implemented training mode")

            trainer.save(save_file)

            logger.debug("Saved results to " + str(save_file))
            logger.debug("--------Finished run" + run + "----------")

            # clean memory after each configuration
            try:
                del trainer
                del batched_output
                del batched_evaluation_output
                del batched_input
            except NameError:
                pass
