import itertools
import random
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


class Player():

    def __init__(self, node_self, graph, current_value):
        self.node_self = node_self
        self.graph = graph
        self.alpha = 2
        self.beta = 3
        self.current_value = current_value

    def evaluate_fully_connected(self, C):
        edge_list = []
        for i in C.keys():
            if C[i]== 1:
                for j in self.graph.adj[i]:
                    edge_list.append((i,j))
                    for k in self.graph.adj[i]:
                        if k > j:
                            edge_list.append((k,j))
        G = nx.from_edgelist(edge_list)
        for i in self.graph.nodes:
            if not G.has_node(i):
                return False
        return nx.is_connected(G)


    def calculate_utility_function(self, C, is_fully_connected):
        # alpha = 2, beta = 3
        # d_i_not_1 = True
        # for i in self.graph.adj[self.node_self]:
        #     if C[i] == 1:
        #         d_i_not_1 = False
        # if d_i_not_1:
        #     h_i = self.beta
        # else:
        #     h_i = 0
        if C[self.node_self] == 0:
            if is_fully_connected:
                g_i = - len(self.graph.nodes) * self.beta
            else:
                g_i = self.beta
        else:
            C_0 = copy.deepcopy(C)
            C_0[self.node_self] = 0
            is_fully_connected_new = self.evaluate_fully_connected(C_0)
            if is_fully_connected_new:
                g_i = - len(self.graph.nodes) * self.beta
            else:
                g_i = self.beta
        if is_fully_connected:
            w_c = 0
        else:
            w_c = -self.alpha
        return w_c,  g_i - self.alpha


    def update_current_value(self, C, is_fully_connected):
        u_0, u_1 = self.calculate_utility_function(C, is_fully_connected)
        current_value = copy.deepcopy(self.current_value)
        if current_value == 0:
            if u_1 > u_0:
                self.current_value = 1
                return True
            return False
        else:
            if u_0 > u_1:
                self.current_value = 0
                return True
            return False



class Game():

    def __init__(self, g, ordering, C = None):
        players = []
        if C == None:
            C = {}
            for i in ordering:
                C[i] = np.random.randint(0,2)
                player_i = Player(i, g, C[i])
                players.append(player_i)
        else:
            for i in ordering:
                player_i = Player(i, g, C[i])
                players.append(player_i)
        self.players = players
        self.C  = C
        self.graph = g
        self.is_fully_connected = False

    def evaluate_fully_connected(self):
        edge_list = []
        for i in self.C.keys():
            if self.C[i]== 1:
                for j in self.graph.adj[i]:
                    edge_list.append((i,j))
                    for k in self.graph.adj[i]:
                        if k > j:
                            edge_list.append((k,j))
        G = nx.from_edgelist(edge_list)
        all_i_in_graph = True
        for i in self.graph.nodes:
            if not G.has_node(i):
                self.is_fully_connected =  False
                all_i_in_graph = False                
        if all_i_in_graph:
            self.is_fully_connected = nx.is_connected(G)

    def run_round(self):
        no_change = True
        for player in self.players:
            has_changed = player.update_current_value(self.C, self.is_fully_connected)
            if has_changed:
                self.C[player.node_self] = player.current_value
                self.evaluate_fully_connected()
                no_change = False
        return no_change

    def run_n_rounds(self, n):
        for i in range(n):
            self.run_round()


    def run_till_nash_equilibrium(self):
        no_change = False
        while not no_change:
            no_change = self.run_round()

    def run_till_nash_equilibrium_or_n_rounds(self, n):
        no_change = False
        i = 0
        while not no_change:
            no_change = self.run_round()
            i += 1
            if i ==n:
                no_change = True


def ordering_based_on_centrality(graph):
    centrality = {}
    for i in graph.nodes:
        centrality[i] = len(graph.adj[i])
    D = []
    j = max(centrality, key=centrality.get)
    D.append(j)
    U_empty = False
    while not U_empty:
        U = set()
        for i in (set(graph.nodes) - set(D)):
            d_i_j = np.infty
            for j in D:
                d_i_curr_j = nx.shortest_path_length(graph, i, j)
                if d_i_curr_j < d_i_j:
                    d_i_j = d_i_curr_j
            if d_i_j == 2:
                U.add(i)
        if len(U) == 0:
            U_empty = True
        else:
            centrality_i = {}
            for i in U:
                centrality_i[i] = centrality[i]
            j = max(centrality_i, key=centrality_i.get)
            D.append(j)
    centrality_rest = {}
    for i in (set(graph.nodes) - set(D)):
        centrality_rest[i] = centrality[i]
    while len(centrality_rest) > 0:
        D.append(max(centrality_rest, key=centrality_rest.get))
        centrality_rest.pop(max(centrality_rest, key=centrality_rest.get))
    return D

def true_centrality_ordering(graph):
    centrality = {}
    D = []
    for i in graph.nodes:
        centrality[i] = len(graph.adj[i])
    while len(centrality) > 0:
        D.append(max(centrality, key=centrality.get))
        centrality.pop(max(centrality, key=centrality.get))
    return D

def evaluate_game_theory_model(node_file_path, edge_file_path, key_dict_path, n, data_storage_location_keep_each_loop= None,data_storage_location_keep_each_loop_simple_model= None ):
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
        ordering = list(graphs[key].nodes)

        game = Game(graphs[key], ordering = ordering)
        game.run_till_nash_equilibrium_or_n_rounds(n)
        print(f"Done with graph: {key}")
        if data_storage_location_keep_each_loop_simple_model != None:
            dictionary = [
                {"Graph key": key, "number_nodes": len(graphs[key].nodes),"objective_value": sum(game.C.values())}]
            dictionary_fieldnames = ["Graph key", "number_nodes","objective_value"]
            if os.path.isfile(data_storage_location_keep_each_loop_simple_model + '.csv'):
                with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writerows(dictionary)
            else:
                with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writeheader()
                    writer.writerows(dictionary)
    # objective_values_optimal = {}
    # if data_storage_location_keep_each_loop != None:
    #     plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
    #     for index, row in plot_information.iterrows():
    #         objective_values_optimal[row["Graph key"]] =  row["objective_value"]
    #
    # objective_values_model = {}
    # if data_storage_location_keep_each_loop_simple_model != None:
    #     plot_information = pd.read_csv(data_storage_location_keep_each_loop_simple_model + ".csv")
    #     for index, row in plot_information.iterrows():
    #         if row["beta"] not in objective_values_model.keys():
    #             objective_values_model[row["beta"]] = {row["Graph key"] : row["objective_value"]}
    #         else:
    #             objective_values_model[row["beta"]][row["Graph key"]]  = row["objective_value"]
    #
    # objective_values = {}
    # for beta in objective_values_model.keys():
    #     for key in objective_values_model[beta].keys():
    #         if key in objective_values_optimal.keys():
    #             if beta not in objective_values.keys():
    #                 objective_values[beta] = [(objective_values_model[beta][key] - objective_values_optimal[key])/  objective_values_optimal[key]]
    #             else:
    #                 objective_values[beta].append((objective_values_model[beta][key] - objective_values_optimal[key])/  objective_values_optimal[key])
    # mean_objectives = {}
    # std_objectives = {}
    # for key in objective_values.keys():
    #     mean_objectives[key] = np.mean(objective_values[key])
    #     std_objectives[key] = np.std(objective_values[key])
    # mean_differences = []
    # std_differences = []
    # # topologies
    # x = []
    # for key in mean_objectives.keys():
    #     mean_differences.append(mean_objectives[key])
    #     std_differences.append(std_objectives[key])
    #     x.append(key)
    # plt.errorbar(x, mean_differences, yerr=std_differences, color="r")
    # plt.xlabel("beta values", fontsize=10)
    # plt.ylabel("Percentage Difference from optimal", fontsize=10)
    # plt.savefig("beta_calculation.png")
    # plt.show()



if __name__ == "__main__":
    evaluate_game_theory_model(node_file_path="3_node_data_different_sizes_no_users.csv",
                                                edge_file_path="3_capacities_different_no_users.csv", key_dict_path="3_key_dict_different_sizes_no_users.csv",
                              n = 100,
                              data_storage_location_keep_each_loop="3_optimal_solution_storage",
                              data_storage_location_keep_each_loop_simple_model="3_game_theory_storage_random_C_O")