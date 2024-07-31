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

class Centrality_Model():

    def __init__(self, graph):
        self.graph = graph

    def degree_centrality(self):
        self.centrality = {}
        for i in self.graph.nodes:
            self.centrality[i] = len(self.graph.adj[i])

    def eccentricity_centrality(self):
        self.centrality = {}
        shortest_paths = nx.shortest_path_length(self.graph)
        for i, shortest_paths_i in shortest_paths:
            shortest_path_length = max(shortest_paths_i.values())
            self.centrality[i] = 1/shortest_path_length

    def closeness_centrality(self):
        self.centrality = {}
        shortest_paths = nx.shortest_path_length(self.graph)
        for i, shortest_paths_i in shortest_paths:
            total_path_length = sum(shortest_paths_i.values())
            self.centrality[i] = 1 / total_path_length

    def closeness_centrality_networkx(self):
        self.centrality = nx.closeness_centrality(self.graph)

    def integrated_path_centrality(self):
        self.centrality = {}
        shortest_paths = nx.shortest_path_length(self.graph)
        c_R = {}
        shortest_paths_dict = {}
        for i, shortest_paths_i in shortest_paths:
            for j in shortest_paths_i.keys():
                shortest_paths_dict[i, j] =shortest_paths_i[j]
            c_R[i] = (len(self.graph.nodes) - sum(shortest_paths_i.values())) / (len(self.graph.nodes)-1)
        delta_g = max(shortest_paths_dict.values())
        additional_term = len(self.graph.nodes) * delta_g / (len(self.graph.nodes) - 1)
        self.centrality = {}
        for i in c_R.keys():
            self.centrality[i] = c_R[i] + additional_term

    def stress_centrality(self):
        self.centrality = {}
        all_shortest_paths = {}
        for j in self.graph.nodes:
            for k in self.graph.nodes:
                all_shortest_paths[j,k] = nx.all_shortest_paths(self.graph, source=j, target=k)
        for i in self.graph.nodes:
            c_s_i = 0
            for j in self.graph.nodes:
                for k in self.graph.nodes:
                    if j != i and k != i:
                        array_of_all_paths = np.array([p for p in all_shortest_paths[j,k]])
                        for path in array_of_all_paths:
                            if i in path:
                                c_s_i += 1
            self.centrality[i] = c_s_i




    def betweenness_centrality(self):
        self.centrality = {}
        all_shortest_paths = {}
        for j in self.graph.nodes:
            for k in self.graph.nodes:
                all_shortest_paths[j,k] = nx.all_shortest_paths(self.graph, source=j, target=k)
        for i in self.graph.nodes:
            c_s_i = 0
            for j in self.graph.nodes:
                for k in self.graph.nodes:
                    if j != i and k != i:
                        sigma_s_t_I = 0
                        array_of_all_paths = np.array([p for p in all_shortest_paths[j,k]])
                        for path in array_of_all_paths:
                            if i in path:
                                sigma_s_t_I += 1
                        sigma_s_t = len(np.array([p for p in all_shortest_paths[j,k]]))
                        if sigma_s_t != 0:
                            c_s_i += sigma_s_t_I/sigma_s_t
            self.centrality[i] = c_s_i

    def betweenness_centrality_network_x(self):
        self.centrality = nx.betweenness_centrality(self.graph)


    def epsilon_betweenness_centrality(self, epsilon):
        self.centrality = {}
        shortest_paths = nx.shortest_path_length(self.graph)
        for i in self.graph.nodes:
            c_s_i = 0
            for j, shortest_paths_j in shortest_paths:
                for k in shortest_paths_j.keys():
                    if j != i and k != i:
                        shortest_paths_len = shortest_paths_j[j]
                        epsilon_betweenness_distance = np.floor((1+epsilon) * shortest_paths_len)
                        paths = nx.all_simple_paths(self.graph, source= j, target = k, cutoff = epsilon_betweenness_distance)
                        sigma_s_t_I = 0
                        array_of_all_paths = np.array([p for p in paths])
                        for path in array_of_all_paths:
                            if i in path:
                                sigma_s_t_I += 1
                        sigma_s_t = len(array_of_all_paths)
                        if sigma_s_t != 0:
                            c_s_i += sigma_s_t_I / sigma_s_t
            self.centrality[i] = c_s_i

    def katz_centrality(self, alpha = 0.1):
        self.centrality = nx.katz_centrality(self.graph, alpha = alpha)

    def bonachich_eigenvector_centrality(self):
        self.centrality = nx.eigenvector_centrality(self.graph, max_iter = 100000)

    def get_detector_set(self):
        D = set()
        j = max(self.centrality, key = self.centrality.get)
        D.add(j)
        U_empty = False
        while not U_empty:
            U= set()
            for i in (set(self.graph.nodes)-D):
                d_i_j = np.infty
                for j in D:
                    d_i_curr_j = nx.shortest_path_length(self.graph, i, j)
                    if d_i_curr_j < d_i_j:
                        d_i_j = d_i_curr_j
                if d_i_j == 2:
                    U.add(i)
            if len(U) == 0:
                U_empty = True
            else:
                centrality = {}
                for i in U:
                    centrality[i] = self.centrality[i]
                j = max(centrality, key=centrality.get)
                D.add(j)
        return D


def evaluate_game_theory_model(node_file_path, edge_file_path, key_dict_path, epsilon, data_storage_location_keep_each_loop= None,data_storage_location_keep_each_loop_simple_model= None ):
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
    current_key = 25
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
        sol_dict, prob = optim.optimisation_detector_placement_run(cmin = 100.0, epsilon = 0.001, N_dmax = 100,  time_limit=1e3)
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
    centralities = ["degree", "betweenness", "eccentricity", "closeness", "integrated", "stress", "Katz", "eigenvector",
                    "epsilon_betweenness"]
    for centrality in centralities:
        if data_storage_location_keep_each_loop_simple_model != None:
            if os.path.isfile(data_storage_location_keep_each_loop_simple_model + f'{centrality}.csv'):
                plot_information = pd.read_csv(data_storage_location_keep_each_loop_simple_model + f'{centrality}.csv')
                last_row_explored = plot_information.iloc[[-1]]
                current_key = last_row_explored["Graph key"].iloc[0]
            else:
                current_key = None
                dictionary_fieldnames = ["Graph key", "number_nodes", "objective_value"]
                with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
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
            cm = Centrality_Model(graphs[key])

            if centrality == "degree":
                cm.degree_centrality()
            elif centrality == "betweenness":
                cm.betweenness_centrality_network_x()
            elif centrality == "eccentricity":
                cm.eccentricity_centrality()
            elif centrality == "closeness":
                cm.closeness_centrality_networkx()
            elif centrality == "integrated":
                cm.integrated_path_centrality()
            elif centrality == "stress":
                cm.stress_centrality()
            elif centrality == "Katz":
                cm.katz_centrality(alpha = 0.1)
            elif centrality == "eigenvector":
                cm.bonachich_eigenvector_centrality()
            elif centrality == "epsilon_betweenness":
                cm.epsilon_betweenness_centrality(epsilon)
            D = cm.get_detector_set()
            print(f"Done with graph: {key}")
            if data_storage_location_keep_each_loop_simple_model != None:
                dictionary = [
                    {"Graph key": key, "number_nodes": len(graphs[key].nodes), "objective_value": len(D)}]
                dictionary_fieldnames = ["Graph key", "number_nodes", "objective_value"]
                if os.path.isfile(data_storage_location_keep_each_loop_simple_model + f'{centrality}.csv'):
                    with open(data_storage_location_keep_each_loop_simple_model + f'{centrality}.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop_simple_model + f'{centrality}.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)


def plot_centrality_models_variation(data_storage_optimal_location, data_storage_centrality_measures, centralities):


    objective_values_optimal = {}
    if data_storage_optimal_location != None:
        plot_information = pd.read_csv(data_storage_optimal_location + ".csv")
        for index, row in plot_information.iterrows():
            objective_values_optimal[row["Graph key"]] =  row["objective_value"]

    objective_values_models = {}
    if data_storage_centrality_measures != None:
        for centrality in centralities:
            plot_information = pd.read_csv(data_storage_centrality_measures + f"{centrality}.csv")
            for index, row in plot_information.iterrows():
                if centrality not in objective_values_models.keys():

                    objective_values_models[centrality] = {row["number_nodes"]:{row["Graph key"] : row["objective_value"]}}
                else:
                    if row["number_nodes"] not in objective_values_models[centrality].keys():
                        objective_values_models[centrality][row["number_nodes"]]  = {row["Graph key"] : row["objective_value"]}#
                    else:
                        objective_values_models[centrality][row["number_nodes"]][row["Graph key"]] = row["objective_value"]

    objective_values = {}
    for centrality in objective_values_models.keys():
        for number_nodes in objective_values_models[centrality].keys():
            for graph_key in objective_values_models[centrality][number_nodes].keys():
                if graph_key in objective_values_optimal.keys():
                    if centrality not in objective_values.keys():
                        objective_values[centrality] = {number_nodes: [(objective_values_models[centrality][number_nodes][graph_key] - objective_values_optimal[graph_key])/  objective_values_optimal[graph_key]]}
                    elif number_nodes not in objective_values[centrality].keys():
                        objective_values[centrality][number_nodes] = [(objective_values_models[centrality][number_nodes][graph_key] - objective_values_optimal[graph_key])/  objective_values_optimal[graph_key]]
                    else:
                        objective_values[centrality][number_nodes].append((objective_values_models[centrality][number_nodes][graph_key] - objective_values_optimal[graph_key])/  objective_values_optimal[graph_key])
    for centrality in objective_values.keys():
        mean_objectives = {}
        std_objectives = {}
        for key in objective_values[centrality].keys():
            mean_objectives[key] = np.mean(objective_values[centrality][key])
            std_objectives[key] = np.std(objective_values[centrality][key])
        mean_differences = []
        std_differences = []
        # topologies
        x = []
        for key in mean_objectives.keys():
            mean_differences.append(mean_objectives[key])
            std_differences.append(std_objectives[key])
            x.append(key)
        plt.errorbar(x, mean_differences, yerr=std_differences, label = f"centrality measure: {centrality}", marker = "o")
    plt.xlabel("Number of Nodes", fontsize=10)
    plt.ylabel("Percentage Difference from optimal", fontsize=10)
    plt.legend()
    plt.savefig("centrality_comparisons_remove_bad_performing_metrics.png")
    plt.show()


if __name__ == "__main__":

    evaluate_game_theory_model(node_file_path="3_node_data_different_sizes_no_users.csv",
                               edge_file_path="3_capacities_different_no_users.csv",
                               key_dict_path="3_key_dict_different_sizes_no_users.csv",
                               epsilon=0.5,
                               data_storage_location_keep_each_loop="3_optimal_solution_storage_2",
                               data_storage_location_keep_each_loop_simple_model="3_centrality_method_storage_")

    # plot_centrality_models_variation(data_storage_optimal_location="3_optimal_solution_storage", data_storage_centrality_measures="3_centrality_method_storage_",
    #                                  centralities = ["degree", "betweenness", "closeness", "integrated", "Katz", "eigenvector"])
    # key_dict, graphs = import_problem_detectors(node_file_path="2_node_data_different_sizes_no_users.csv",
    #                                             edge_file_path="2_capacities_different_no_users.csv", key_dict_path="2_key_dict_different_sizes_no_users.csv")
    # for key in graphs.keys():
    #     cm = Centrality_Model(graphs[key])
    #     cm.betweenness_centrality_network_x()
    #     D = cm.get_detector_set()
    #     print(f"Graph {key} Detectors {len(D)}")