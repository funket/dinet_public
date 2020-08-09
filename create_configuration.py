import configparser
import os
import argparse

import dinet_base as dinet

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execution learning")
    parser.add_argument("--configfile", default="configuration.ini",
                        help="select configuration file default configuration.ini ")
    parser.add_argument("--gpu", default=-1, type=int)
    args = parser.parse_args()

    print("Creating " + args.configfile + " and folders for data")

    config = configparser.ConfigParser()

    sample_size = 10
    number_of_batch_sizes = [10]
    file = "toy_example_5_groups"
    # file = "out.moreno_blogs_blogs"
    # file = "out.subelj_cora_cora"
    # file = "out.cit_HepTh"
    # file = "out.petster_hamster"
    optimizers = [
        "Adam",
    ]

    exponents = [
        1 / 2,
        # 1,
        # 1 / 3,
        # 1 / 4
    ]

    training_modes = [
        "corrected",
        "mixed",
        # "uncorrected",
    ]

    run_counter = 0

    config["global"] = {
        # "file": "data/" + file,
        "max_batches": max(number_of_batch_sizes),
        "sample_size": 10,
        "training_data_folder": "training_data",
        "global_folder": "toy_example_5_groups",
        "gpu_number": "1",
        "number_of_loops": 150,
        "number_of_epochs": 1,
        "embedding_dimension": 2,
        "directed_graph": True,
        "evaluate_with_all_data": False,
        "stochastic_training": True,
    }

    if args.gpu != -1:
        config["global"]["gpu_number"] = str(args.gpu)

    # create folders
    global_folder = "./" + config["global"]["global_folder"]
    os.mkdir(global_folder)
    global_folder = global_folder + "/"
    os.mkdir(global_folder + "results")
    os.mkdir(global_folder + config["global"]["training_data_folder"])

    original_file = "data/" + file
    file_paths = [("data/" + file, file)]


    def add_single_configuration():
        global run_counter
        global config
        training_rates = [
            .1,
            # .05,
            # .25,
            # .01,
        ]
        for training_rate in training_rates:
            single_configuration["learning_rate"] = training_rate
            for optimizer in optimizers:
                single_configuration["optimizer"] = optimizer
                for exponent in exponents:
                    single_configuration["exponent"] = exponent

                    for training_mode in training_modes:
                        single_configuration["training_mode"] = training_mode

                        # option to create multiple runs
                        for i in range(1):
                            # write single configuration
                            for i, (file_path, file_short) in enumerate(file_paths):
                                single_configuration["file"] = file_path
                                single_configuration[
                                    "save_file"] = global_folder + "results/" + file_short + "_save_" + str(
                                    run_counter) + ".npz"
                                single_configuration[
                                    "evaluation_save_file"] = global_folder + "results/" + file_short + "_evaluation_" + str(
                                    run_counter) + ".npz"

                                config[run_counter * len(file_paths) + i] = single_configuration
                            run_counter += 1


    # if possible use all edges
    able_to_use_all_pairs = True

    if able_to_use_all_pairs:
        single_configuration = {"all_pairs": able_to_use_all_pairs,
                                "number_of_batches": 1  # 3690  # 1  # 153  # 2777 #  410
                                }

        add_single_configuration()

    else:
        # start with close
        single_configuration = {"number_of_close": sample_size}

        # add infinities
        single_configuration["number_of_infinities"] = sample_size
        for number_of_batches in number_of_batch_sizes:
            single_configuration["number_of_batches"] = number_of_batches

            add_single_configuration()


    with open(args.configfile, 'w') as configfile:
        config.write(configfile)

    with open(global_folder + '/configuration.ini', 'w') as configfile:
        config.write(configfile)
