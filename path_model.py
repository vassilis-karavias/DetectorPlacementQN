import itertools

import numpy as np
import networkx as nx
from itertools import islice
import copy
import pandas as pd
from import_graph_data import import_problem_detectors
import cplex
import os
import csv
import optimisation_detector_no_switching
import matplotlib.pyplot as plt

class Path_Simple():

    def __init__(self, n, graph, key_dict, prob_exp_model = False):
        """
        Path Evaluation method class. Probability method evaluations can be selected.
        Parameters
        ----------
        n: number of shortest paths to use between nodes
        graph: graph of nodes
        key_dict: key_dict of keys required between user connections
        """
        self.n = n
        self.graph = graph
        self.key_dict = key_dict
        self.key_dict_temp = copy.deepcopy(key_dict)
        self.nodes_added = []
        self.prob_exp_model = prob_exp_model
    def _n_shortest_paths(self, source, target):
        return list(islice(nx.shortest_simple_paths(self.graph, source, target),self.n))

    def _get_lists_of_shortest_paths(self):
        paths = {}
        for source, target in self.key_dict_temp:
            paths[source, target] = self._n_shortest_paths(source, target)
        return paths

    def calculate_weight_of_path_reciprocal(self, path):
        length = 0
        for i in range(len(path)):
            if i == 0:
                if path[i] not in self.nodes_added and path[i + 1] not in self.nodes_added:
                    length += 1
            elif i == len(path) - 1:
                if path[i] not in self.nodes_added and path[i - 1] not in self.nodes_added:
                    length += 1
            else:  #
                if path[i] not in self.nodes_added and path[i - 1] not in self.nodes_added and path[
                    i + 1] not in self.nodes_added:
                    length += 1
        assert length > 0, f"cost is 0 for path {path}"
        return 1/length

    def calculate_weight_of_path_exponential(self, path, beta):
        length = 0
        for i in range(len(path)):
            if i == 0:
                if path[i] not in self.nodes_added and path[i + 1] not in self.nodes_added:
                    length += 1
            elif i == len(path) - 1:
                if path[i] not in self.nodes_added and path[i - 1] not in self.nodes_added:
                    length += 1
            else:  #
                if path[i] not in self.nodes_added and path[i - 1] not in self.nodes_added and path[
                    i + 1] not in self.nodes_added:
                    length += 1
        return np.exp(-beta * length)


    def calculate_prob_path(self, paths_list, beta = None):
        costs_each_path = []
        for path in paths_list:
            if self.prob_exp_model:
                costs_each_path.append(self.calculate_weight_of_path_exponential(path, beta))
            else:
                costs_each_path.append(self.calculate_weight_of_path_reciprocal(path))
        total_costs = sum(costs_each_path)
        costs_total = []
        for cost in costs_each_path:
            costs_total.append(cost/total_costs)
        return costs_total


    def calculate_utility_node(self, node, paths_dict, beta = None):
        weight = 0.0
        for source, target in paths_dict.keys():
            costs = self.calculate_prob_path(paths_dict[source,target], beta)
            for i in range(len(paths_dict[source,target])):
                if node in paths_dict[source,target][i]:
                    weight += costs[i]
        return weight



    def full_recursion(self, beta = None):
        while len(self.key_dict_temp.keys()) > 0:
            # calculate shortest paths
            paths = self._get_lists_of_shortest_paths()
            # Evaluate importance of nodes
            weights = {}
            for node in self.graph.nodes():
                if node not in self.nodes_added:
                    weights[node] = self.calculate_utility_node(node, paths, beta)
            node_to_add = max(weights, key=weights.get)
            # add most important node to the set of detector nodes
            self.nodes_added.append(node_to_add)
            # evaluate if adding the node has made any of the paths cost 0 for any key_dict
            # if it has, this means these paths are connected and do not need further analysis
            # Here we can also look at if the path has sufficient capacity availability and
            # analyse this but we ignore this aspect in this model
            key_dict_temp = copy.deepcopy(self.key_dict_temp)
            for keys in self.key_dict_temp.keys():
                for path in paths[keys]:
                    length = 0
                    for i in range(len(path)):
                        if i ==0:
                            if path[i] not in self.nodes_added and path[i+1] not in self.nodes_added:
                                length +=1
                        elif i == len(path) - 1:
                            if path[i] not in self.nodes_added and path[i-1] not in self.nodes_added:
                                length +=1
                        else:#
                            if path[i] not in self.nodes_added and path[i-1] not in self.nodes_added and path[i+1] not in self.nodes_added:
                                length +=1
                    if length <= 0.001:
                        key_dict_temp.pop(keys, None)
            self.key_dict_temp = key_dict_temp

class Postprocessing():

    def __init__(self, graph, key_dict, n, nodes_added):
        self.n = n
        self.graph = graph
        self.key_dict = key_dict
        self.nodes_added = nodes_added


    def calculate_length_path(self, path, nodes_added):
        length = 0
        for i in range(len(path)):
            if i == 0:
                if path[i] not in nodes_added and path[i + 1] not in nodes_added:
                    length += 1
            elif i == len(path) - 1:
                if path[i] not in nodes_added and path[i - 1] not in nodes_added:
                    length += 1
            else:  #
                if path[i] not in nodes_added and path[i - 1] not in nodes_added and path[
                    i + 1] not in nodes_added:
                    length += 1
        return length


    def _n_shortest_paths(self, source, target):
        return list(islice(nx.shortest_simple_paths(self.graph, source, target),self.n))
    def _get_lists_of_shortest_paths(self):
        paths = {}
        for source, target in self.key_dict:
            paths[source, target] = self._n_shortest_paths(source, target)
        return paths


    def path_set_zero_not_empty(self, nodes_added):
        path_set_not_empty = {}
        paths = self._get_lists_of_shortest_paths()
        for keys in self.key_dict.keys():
            path_set_not_empty[keys] = None
            path_set_not_empty[keys] = False
            for path in paths[keys]:
                length = self.calculate_length_path(path, nodes_added)
                if length < 0.01:
                    path_set_not_empty[keys] = True
        return all(path_set_not_empty.values())


    def get_smallest_true_set(self):
        nodes_added_possible_list = []
        for l in range(1, len(self.nodes_added) + 1):
            for subset in itertools.combinations(self.nodes_added, l):
                if self.path_set_zero_not_empty(subset):
                    return subset, l


class NodeSelection(Postprocessing):

    def __init__(self, graph, key_dict, n):
        super().__init__( graph, key_dict, n, nodes_added= graph.nodes)



def evaluate_values_with_beta(node_file_path, edge_file_path, key_dict_path, n, data_storage_location_keep_each_loop= None,data_storage_location_keep_each_loop_simple_model= None ):
    key_dict, graphs = import_problem_detectors(node_file_path=node_file_path,
                                                edge_file_path=edge_file_path,
                                                key_dict_path=key_dict_path)

    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
    for key in graphs.keys():
        if current_key != None:
            if current_key == key:
                current_key = None
                continue
            else:
                continue
        prob = cplex.Cplex()
        key_dict_bidirect = optimisation_detector_no_switching.make_key_dict_bidirectional(key_dict[key])
        optim = optimisation_detector_no_switching.No_Switching_Detector_Optimisation(prob = prob, g = graphs[key], key_dict = key_dict_bidirect)
        sol_dict, prob = optim.optimisation_detector_placement_run(cmin = 100.0, epsilon = 0.001, N_dmax = 10,  time_limit=1e2)
        if data_storage_location_keep_each_loop != None:
            dictionary = [
                {"Graph key": key, "objective_value": prob.solution.get_objective_value()}]
            dictionary_fieldnames = ["Graph key", "objective_value"]
            if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writerows(dictionary)
            else:
                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writeheader()
                    writer.writerows(dictionary)

    if data_storage_location_keep_each_loop_simple_model != None:
        if os.path.isfile(data_storage_location_keep_each_loop_simple_model + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_simple_model + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["beta"]
            current_key = last_row_explored["Graph key"].iloc[0]
            beta_current = last_ratio_done.iloc[0]
        else:
            current_key = None
            beta_current = None
            dictionary_fieldnames = ["beta","Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
        beta_current = None

    for beta in np.arange(0, 10, 0.1):
        if beta_current != None:
            if beta != beta_current:
                continue
            elif beta == beta_current:
                beta_current = None
        for key in graphs.keys():
            if current_key != None:
                if current_key == key:
                    current_key = None
                    continue
                else:
                    continue
            path_eval = Path_Simple(n=n, graph=graphs[key], key_dict=key_dict[key], prob_exp_model=True)
            path_eval.full_recursion(beta=beta)
            print(f"Done with beta: {beta}, graph: {key}")
            if data_storage_location_keep_each_loop_simple_model != None:
                dictionary = [
                    {"beta": beta, "Graph key": key, "objective_value": len(path_eval.nodes_added)}]
                dictionary_fieldnames = ["beta","Graph key", "objective_value"]
                if os.path.isfile(data_storage_location_keep_each_loop_simple_model + '.csv'):
                    with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)
    objective_values_optimal = {}
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            objective_values_optimal[row["Graph key"]] =  row["objective_value"]

    objective_values_model = {}
    if data_storage_location_keep_each_loop_simple_model != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_simple_model + ".csv")
        for index, row in plot_information.iterrows():
            if row["beta"] not in objective_values_model.keys():
                objective_values_model[row["beta"]] = {row["Graph key"] : row["objective_value"]}
            else:
                objective_values_model[row["beta"]][row["Graph key"]]  = row["objective_value"]

    objective_values = {}
    for beta in objective_values_model.keys():
        for key in objective_values_model[beta].keys():
            if key in objective_values_optimal.keys():
                if beta not in objective_values.keys():
                    objective_values[beta] = [(objective_values_model[beta][key] - objective_values_optimal[key])/  objective_values_optimal[key]]
                else:
                    objective_values[beta].append((objective_values_model[beta][key] - objective_values_optimal[key])/  objective_values_optimal[key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r")
    plt.xlabel("beta values", fontsize=10)
    plt.ylabel("Percentage Difference from optimal", fontsize=10)
    plt.savefig("beta_calculation.png")
    plt.show()





if __name__ == "__main__":

    evaluate_values_with_beta(node_file_path="2_node_data_different_sizes_no_users.csv",
                                                edge_file_path="2_capacities_different_no_users.csv", key_dict_path="2_key_dict_different_sizes_no_users.csv",
                              n = 100,
                              data_storage_location_keep_each_loop="2_optimal_solution_storage",
                              data_storage_location_keep_each_loop_simple_model="2_alpha_variation_storage")
    # key_dict, graphs = import_problem_detectors(node_file_path="1_node_data_different_sizes_no_users.csv",
    #                                             edge_file_path="1_capacities_different_no_users.csv", key_dict_path="1_key_dict_different_sizes_no_users.csv")
    # for key in graphs.keys():
    #     path_eval = Path_Simple(n = 100, graph = graphs[key], key_dict =key_dict[key], prob_exp_model = True)
    #     path_eval.full_recursion(beta = 1)
    #     print(f"Graph {key}, number of detectors: "+ str(len(path_eval.nodes_added)))
    #     post_process = Postprocessing(graph = graphs[key], key_dict = key_dict[key], n = 100, nodes_added = path_eval.nodes_added)
    #     nodes_final, l = post_process.get_smallest_true_set()
    #     print(f"After Postprocessing: Graph {key}, number of detectors: "+ str(l))
    #     node_selection = NodeSelection(graph = graphs[key], key_dict = key_dict[key], n = 100)
    #     nodes_final, l = node_selection.get_smallest_true_set()
    #     print(f"Node Selection Brute Force Method: Graph {key}, number of detectors: " + str(l))
