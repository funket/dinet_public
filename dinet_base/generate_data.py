# Load the graph and generate the input for the learning of the model
# The results are
#  1. a two dimensional list of edges, e.g.
#    edges = [[0, 1, 2, 4], [3, 2, 1, 0]]
#   for the edges (0,3), (1,2), (2,1) and (4,0)
#
#  2. a list of the same length with the distances of these edges, e.g.
#    distances [1, 2, inf, 1]
#   (infinity is allowed)
#
# For this task we offer different alternatives, which can be chosen according
# to the presented results in our publication:
#
#  0. All pairs shortest paths (space needed N^2), only advisable for small/medium size networks
#  1. K local neighbors
#  2. K unreachable nodes (i.e. infinity sampling)

import networkx as nx
import numpy as np
import logging
import random

logger = logging.getLogger("dinet")


def remove_diagonal_and_flatten(matrix, dimension):
    s0, s1 = matrix.strides
    return np.lib.stride_tricks.as_strided(matrix.ravel()[1:],
                                           shape=(dimension - 1, dimension),
                                           strides=(s0 + s1, s1)).flatten()


# 0. All pairs shortest paths
def get_all_pairs_shortest_path(graph, as_full_distance_matrix=False):
    all_distances = nx.all_pairs_shortest_path_length(graph)

    number_of_nodes = len(graph)

    raw_distances = np.zeros(shape=(number_of_nodes, number_of_nodes))
    raw_distances[raw_distances == 0] = np.inf

    for from_node, paths in all_distances:
        for to_node, length in paths.items():
            raw_distances[(from_node, to_node)] = length

    if as_full_distance_matrix:
        return raw_distances

    edges = np.zeros(shape=(2, number_of_nodes * (number_of_nodes - 1)), dtype=np.int32)

    distances = remove_diagonal_and_flatten(raw_distances, number_of_nodes)
    edges[0, :] = remove_diagonal_and_flatten(
        np.linspace(np.zeros(number_of_nodes), np.full(number_of_nodes, number_of_nodes - 1), number_of_nodes),
        number_of_nodes)
    edges[1, :] = remove_diagonal_and_flatten(
        np.linspace(np.arange(number_of_nodes), np.arange(number_of_nodes), number_of_nodes), number_of_nodes)

    return edges, distances


# helper function for local neighborhood
def limited_breadth_first_search(neighbor_iterator, node, check_new_value, number_of_close):
    # initialize with all direct neighbors
    close_to_node = []
    seen_breadth_first = set()

    number_of_new_found_nodes = 0
    for neighbor in neighbor_iterator(node):
        if number_of_new_found_nodes == number_of_close:
            break

        seen_breadth_first.add(neighbor)
        close_to_node.append(neighbor)

        if check_new_value(neighbor):
            number_of_new_found_nodes += 1
    # which have the distance 1
    number_of_found_nodes = len(close_to_node)
    distances = [1] * number_of_found_nodes

    current_visiting_node_index = 0
    while current_visiting_node_index < number_of_found_nodes and number_of_new_found_nodes < number_of_close:
        new_nodes = 0

        for neighbor in neighbor_iterator(close_to_node[current_visiting_node_index]):
            if neighbor not in seen_breadth_first:
                seen_breadth_first.add(neighbor)
                close_to_node.append(neighbor)
                new_nodes += 1
                if check_new_value(neighbor):
                    number_of_new_found_nodes += 1
                if number_of_new_found_nodes == number_of_close:
                    break

        distances.extend([distances[current_visiting_node_index] + 1] * new_nodes)

        number_of_found_nodes = len(close_to_node)
        current_visiting_node_index += 1

    return distances, close_to_node


# 1. K local neighbors
def get_local_neighborhood(graph, number_of_close=10, batches=1, exclude_nodes=None):
    # track which edges are already known
    handled_edges = set()
    edge_information = []

    if exclude_nodes is None:
        exclude_nodes = set()
    else:
        exclude_nodes = set(exclude_nodes)

    # for each node read local neighborhood
    for node in graph:

        if node in exclude_nodes:
            # skip nodes
            continue

        distances, close_to_node = limited_breadth_first_search(
            graph.successors,
            node,
            lambda neighbor: (neighbor, node) not in handled_edges and neighbor not in exclude_nodes,
            (batches + 1) * number_of_close)

        # close incoming
        incoming_distances, incoming_close_to_node = limited_breadth_first_search(
            graph.predecessors,
            node,
            lambda neighbor: (node, neighbor) not in handled_edges and neighbor not in exclude_nodes,
            (batches + 1) * number_of_close)

        # produce half half results
        handled_successors = 0
        open_successors = len(close_to_node)
        handled_predecessors = 0
        open_predecessors = len(incoming_close_to_node)
        for batch in range(batches):
            added = 0
            i = 0
            j = 0
            for i in range(open_successors):
                open_successors -= 1

                close_node = close_to_node[handled_successors + i]
                if (node, close_node) not in handled_edges:
                    added += 1
                    handled_edges.add((node, close_node))
                    edge_information.append((batch, distances[handled_successors + i]))

                if added >= number_of_close / 2:
                    break
            handled_successors += i

            for j in range(open_predecessors):
                open_predecessors -= 1

                close_node = incoming_close_to_node[handled_predecessors + j]
                if (close_node, node) not in handled_edges:
                    added += 1
                    handled_edges.add((close_node, node))
                    edge_information.append((batch, incoming_distances[handled_predecessors + j]))

                if added == number_of_close:
                    break
            handled_predecessors += j

            for i in range(open_successors):
                open_successors -= 1
                close_node = close_to_node[handled_successors + i]
                if (node, close_node) not in handled_edges:
                    added += 1
                    handled_edges.add((node, close_node))
                    edge_information.append((batch, distances[handled_successors + i]))

                if added == number_of_close:
                    break
            handled_successors += i

    inputs = [[[], []] for _ in range(batches)]
    outputs = [[] for _ in range(batches)]
    for (from_node, to_node), (batch, distance) in zip(handled_edges, edge_information):
        inputs[batch][0].append(from_node)
        inputs[batch][1].append(to_node)
        outputs[batch].append(distance)

    return inputs, outputs


class Tarjan:

    def __init__(self):
        self.current_index = 0
        self.index = np.full(1, -1)
        self.low_link = np.full(1, -1)
        self.on_stack = np.full(1, False)
        self.stack = []

        self.i = 0

        self.scc_counter = 0
        self.strongly_connected_components = []
        self.scc_successors = []
        self.number_of_members_in_scc = []
        self.reachable_scc = []
        self.identified_nodes = 0
        self.nodes_by_scc_in_reversed_topological_order = np.full(1, -1)

    def perform_networkx_tarjan(self, graph):
        self.number_of_members_in_scc = []
        number_of_nodes = len(graph)
        self.index = np.full(number_of_nodes, -1)
        self.nodes_by_scc_in_reversed_topological_order = np.full(number_of_nodes, -1)
        identified_node = -1

        for current_number_of_scc, scc in enumerate(nx.strongly_connected_components(graph)):
            self.number_of_members_in_scc.append(len(scc))
            for identified_node, node in enumerate(scc, start=identified_node + 1):
                self.index[node] = current_number_of_scc
                self.nodes_by_scc_in_reversed_topological_order[identified_node] = node

    def get_infinities(self, graph, number_of_infinities, batches=1, sample_random_inf="random"):
        self.perform_networkx_tarjan(graph)
        number_of_nodes = len(graph)

        logger.debug("Finished calculating SCC, now starting sampling")

        # use nodes in reversed topological order to sample infinities
        lower_bound = 0
        last_scc = -1

        inputs = [[[], []] for _ in range(batches)]

        for i, node in enumerate(self.nodes_by_scc_in_reversed_topological_order):
            scc = self.index[node]
            if scc != last_scc:
                lower_bound += self.number_of_members_in_scc[scc]
                last_scc = scc

            number_of_selected_nodes = min(number_of_nodes - lower_bound, batches * number_of_infinities)
            if sample_random_inf == "numpy_random":
                to_infinities = np.random.choice(self.nodes_by_scc_in_reversed_topological_order[lower_bound:],
                                                 number_of_selected_nodes, replace=False)
            elif sample_random_inf == "pseudo":
                to_infinities = np.random.choice(
                    self.nodes_by_scc_in_reversed_topological_order[
                    lower_bound:lower_bound + 10 * batches * number_of_infinities],
                    number_of_selected_nodes, replace=False)
            elif sample_random_inf == "random":
                to_infinities = self.nodes_by_scc_in_reversed_topological_order[lower_bound:][
                    random.sample(range(number_of_nodes - lower_bound), number_of_selected_nodes)]
            else:
                to_infinities = self.nodes_by_scc_in_reversed_topological_order[
                                lower_bound:lower_bound + number_of_selected_nodes]

            number_of_found_infinities = len(to_infinities)
            for batch in range(batches):
                inputs[batch][1].extend(
                    to_infinities[batch * number_of_infinities:(batch + 1) * number_of_infinities])
                if (batch + 1) * number_of_infinities < number_of_found_infinities:
                    inputs[batch][0].extend([node] * number_of_infinities)
                else:
                    inputs[batch][0].extend([node] * (number_of_found_infinities - batch * number_of_infinities))
                    # nothing to add in further steps
                    break

            if i % 10000 == 0:
                logger.debug("Finished " + str(i) + " nodes")

        outputs = [[np.inf] * len(inputs[batch][0]) for batch in range(batches)]

        return inputs, outputs


# 2. K unreachable nodes
def sample_infinities(graph, number_of_infinities=10, batches=1, known_distances=None):
    inputs = [[], []]
    outputs = []
    # read infinities
    if known_distances is not None:
        if batches > 1:
            raise NotImplementedError()

        values = set()
        for node in graph:
            to_inf = np.where(known_distances[node, :] == np.inf)[0]
            from_inf = np.where(known_distances[:, node] == np.inf)[0]

            to_inf = np.random.permutation(to_inf)
            from_inf = np.random.permutation(from_inf)

            added = 0
            for i, to_node in enumerate(to_inf):
                if added >= number_of_infinities / 2:
                    break
                if (node, to_node, np.inf) not in values:
                    added += 1
                    values.add((node, to_node, np.inf))

            for j, from_node in enumerate(from_inf):
                if added == number_of_infinities:
                    break
                if (from_node, node, np.inf) not in values:
                    added += 1
                    values.add((from_node, node, np.inf))

        for from_node, to_node, distance in values:
            inputs[0].append(from_node)
            inputs[1].append(to_node)
            outputs.append(distance)
            return inputs, outputs
    else:
        # use topological sorting of strongest connected component
        # based on Tarjan's algorithm
        return Tarjan().get_infinities(graph, number_of_infinities, batches)


def breadth_first_search(neighbor_iterator, node, number_of_nodes):
    # initialize with all direct neighbors
    neighbors_of_this_node = []
    visited = np.zeros(number_of_nodes)
    # mark starting node as visited
    visited[node] = 1

    for neighbor in neighbor_iterator(node):
        visited[neighbor] = 1
        neighbors_of_this_node.append(neighbor)
    # which have the distance 1
    distances = [1] * len(neighbors_of_this_node)

    current_visiting_node_index = 0
    while current_visiting_node_index < len(neighbors_of_this_node):

        new_nodes = 0
        for neighbor in neighbor_iterator(neighbors_of_this_node[current_visiting_node_index]):
            # if not already visited
            if visited[neighbor] == 0:
                visited[neighbor] = 1
                neighbors_of_this_node.append(neighbor)
                new_nodes += 1

        distances.extend([distances[current_visiting_node_index] + 1] * new_nodes)
        current_visiting_node_index += 1

    return neighbors_of_this_node, distances, visited


def read_graph_from_file(file_path: str, is_directed=True, label_attribute=None):
    """
    Creates graph from edge list. Convert node labels to 0, ..., N-1 and removes selfloops
    :param file_path:
    :param is_directed:
    :param label_attribute:
    :return: 
    """
    if is_directed:
        graph = nx.read_edgelist(file_path, comments="%", nodetype=int, create_using=nx.DiGraph(), data=False)
    else:
        graph = nx.read_edgelist(file_path, comments="%", nodetype=int, create_using=nx.Graph(), data=False)
        graph = nx.DiGraph(graph)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute=label_attribute)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


# Aggregating different strategies into single function
def generate_input(graph,
                   calculate_all_pairs=False,
                   number_of_close=10,
                   number_of_infinities=10,
                   batches=1,
                   ):
    if calculate_all_pairs:
        raw_inputs, raw_outputs = get_all_pairs_shortest_path(graph)
        logger.debug("Read all distances now transforming")
        raw_inputs = raw_inputs.reshape((2, batches, -1))
        batched_input = [np.array(raw_inputs[:, batch, :], dtype=np.int) for batch in range(batches)]
        raw_outputs = raw_outputs.reshape(batches, -1)
        batched_output = [np.array(raw_outputs[batch]) for batch in range(batches)]
        return batched_input, batched_output, np.arange(len(graph))

    raw_inputs = [[[], []] for _ in range(batches)]
    raw_outputs = [[] for _ in range(batches)]

    if number_of_close > 0:
        close_input, close_output = get_local_neighborhood(graph, number_of_close, batches)

        for batch in range(batches):
            raw_inputs[batch][0].extend(close_input[batch][0])
            raw_inputs[batch][1].extend(close_input[batch][1])

            raw_outputs[batch].extend(close_output[batch])

            # check length
            if not (len(raw_inputs[batch][0]) == len(raw_inputs[batch][1]) == len(raw_outputs[batch])):
                raise ValueError("Input sizes of " + str(batch) + " batch of close neighborhood varies:"
                                 + " From edges " + str(len(close_input[batch][0]))
                                 + " to edges " + str(len(close_input[batch][1]))
                                 + " distances " + str(len(close_output[batch]))
                                 )

        logger.debug(
            "Finished reading close: inputs" + " ".join([str(len(inputs[0])) for inputs in close_input])
            + " output " + " ".join([str(len(outputs)) for outputs in close_output])
        )

        del close_input
        del close_output

    if number_of_infinities > 0:
        inf_input, inf_output = sample_infinities(graph, number_of_infinities, batches)

        for batch in range(batches):
            raw_inputs[batch][0].extend(inf_input[batch][0])
            raw_inputs[batch][1].extend(inf_input[batch][1])

            raw_outputs[batch].extend(inf_output[batch])

            # check length
            if not (len(raw_inputs[batch][0]) == len(raw_inputs[batch][1]) == len(raw_outputs[batch])):
                raise ValueError("Input sizes of " + str(batch) + " batch of close neighborhood varies:"
                                 + " From edges " + str(len(inf_input[batch][0]))
                                 + " to edges " + str(len(inf_input[batch][1]))
                                 + " distances " + str(len(inf_output[batch]))
                                 )

        logger.debug(
            "Finished reading inf: inputs" + " ".join([str(len(inputs[0])) for inputs in inf_input])
            + " output " + " ".join([str(len(outputs)) for outputs in inf_output])
        )

        del inf_input
        del inf_output

    # return tf.constant(raw_inputs), tf.cast(raw_outputs, tf.float32)
    return ([np.array(raw_inputs[batch], dtype=np.int) for batch in range(batches)],
            [np.array(raw_outputs[batch]) for batch in range(batches)],
            np.array(np.array(1))
            )


def save_training_data(file, batched_input, batched_output, landmarks):
    enumerated_numpy_arrays = {"landmarks": landmarks}
    for i, input_array in enumerate(batched_input):
        enumerated_numpy_arrays["input_" + str(i)] = input_array

    for i, output_array in enumerate(batched_output):
        enumerated_numpy_arrays["output_" + str(i)] = output_array

    np.savez_compressed(file, **enumerated_numpy_arrays)


def load_training_data(file, number_of_batches):
    variables = np.load(file)

    landmarks = variables["landmarks"]
    batched_input = []
    batched_output = []

    for i in range(number_of_batches):
        batched_input.append(variables["input_" + str(i)])
        batched_output.append(variables["output_" + str(i)])

    return batched_input, batched_output, landmarks
