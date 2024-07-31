import cplex
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import os
import csv
import copy
import networkx as nx
from import_graph_data import import_problem_detectors


def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names=prob.variables.get_names()
    values=prob.solution.get_values()
    sol_dict={names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict




class Detector_Optimisation():

    def __init__(self, prob, g, key_dict):
        self.prob = prob
        self.g = g
        self.key_dict = key_dict

    def _objective_value(self):
        obj_vals = []
        for i in self.g.nodes:
            obj_vals.append((f"delta_{i}", 1))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def _flow_conservation(self):
        """
         Add the conservation of flow constraint:

        \sum_{m \in N(n)} x^{k}_{(n,m)} + x^{k_R}_{(m,n)} - x^{k}_{(m,n)} - x^{k_R}_{(n,m)} = 0
        """
        for i in self.g.nodes:
            for k in self.key_dict:
                if k[0] < k[1] and k[1] != i and k[0] != i:
                    flow = []
                    val = []
                    for n in self.g.neighbors(i):
                        flow.extend(
                            [f"x{i}_{n}_k{k[0]}_{k[1]}", f"x{n}_{i}_k{k[1]}_{k[0]}", f"x{n}_{i}_k{k[0]}_{k[1]}",
                             f"x{i}_{n}_k{k[1]}_{k[0]}"])
                        val.extend([1, 1, -1, -1])
                    lin_expressions = [cplex.SparsePair(ind=flow, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["E"], rhs=[0.])

    def _no_flow_out_sink_in_source(self):
        for k in self.key_dict:
            if k[0] < k[1]:
                for n in self.g.neighbors(k[1]):
                    ind = [f"x{k[1]}_{n}_k{k[0]}_{k[1]}", f"x{n}_{k[1]}_k{k[1]}_{k[0]}"]
                    val = [1, 1]
                    lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["E"], rhs=[0])
                for n in self.g.neighbors(k[0]):
                    ind = [f"x{n}_{k[0]}_k{k[0]}_{k[1]}", f"x{k[0]}_{n}_k{k[1]}_{k[0]}"]
                    val = [1, 1]
                    lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["E"], rhs=[0])

    def _connectivity_constraint(self, cmin):
        for k in self.key_dict:
            if k[0] < k[1]:
                ind = []
                val = []
                for n in self.g.neighbors(k[1]):
                    ind.extend([f"x{n}_{k[1]}_k{k[0]}_{k[1]}", f"x{k[1]}_{n}_k{k[1]}_{k[0]}"])
                    val.extend([1, 1])
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                if isinstance(cmin, dict):
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["G"],
                                                     rhs=[float(self.key_dict[k] * cmin[k])])
                else:
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["G"],
                                                     rhs=[float(self.key_dict[k] * cmin)])

    def _ensure_enough_detectors_on_node(self, *args, **kwargs):
        pass


class No_Switching_Detector_Optimisation(Detector_Optimisation):

    def __init__(self, prob, g, key_dict):
        super().__init__(prob=prob, g=g, key_dict=key_dict)



    def _ensure_enough_detectors_on_node(self, epsilon, N_dmax):
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        y_var = []
        for edge in self.g.edges:
            y_var.append(f"y_{edge[0],edge[1]}")
        self.prob.variables.add(names = y_var, types = [self.prob.variables.type.integer] * len(y_var))
        deltas = []
        for i in self.g.nodes:
            deltas.append(f"delta_{i}")
        self.prob.variables.add(names = deltas, types = [self.prob.variables.type.binary] * len(deltas))
        for i,j in self.g.edges:
            var_g = [f"y_{i,j}"]
            val_g = [1]
            var_l = [f"y_{i,j}"]
            val_l = [1]
            capacity = int(self.g.edges[[i, j]]["capacity"])
            if capacity == 0:
                constraint = [cplex.SparsePair(ind = var_g, val = val_g)]
                self.prob.linear_constraints.add(lin_expr=constraint, senses=["L"],
                                                 rhs=[0.0])
                var_zero = []
                val_zero = []
                for k in self.key_dict:
                    var_zero.extend([f"x{j}_{i}_k{k[0]}_{k[1]}"])
                    val_zero.extend([1])
                constraint = [cplex.SparsePair(ind = var_zero, val = val_zero)]
                self.prob.linear_constraints.add(lin_expr=constraint, senses=["L"],
                                                 rhs=[0.0])
            else:
                for k in self.key_dict:
                    var_g.extend([f"x{j}_{i}_k{k[0]}_{k[1]}"])
                    val_g.extend([-1/capacity])
                    var_l.extend([f"x{j}_{i}_k{k[0]}_{k[1]}"])
                    val_l.extend([-1 / capacity])
                constraints = [cplex.SparsePair(ind = var_g, val = val_g), cplex.SparsePair(ind = var_l, val = val_l)]
                self.prob.linear_constraints.add(lin_expr = constraints, senses = ["GL"], rhs = [0.0, 1 + epsilon])
        for i in self.g.nodes:
            ind = []
            val = []
            for j in self.g.adj[i]:
                ind.extend([f"y_{i,j}"])
                val.extend([1])
            ind.extend([f"delta_{i}"])
            val.extend([-N_dmax])
            contraint = [cplex.SparsePair(ind = ind, val = val)]
            self.prob.linear_constraints.add(lin_expr = contraint, senses = ["L"], rhs  = [0.0])



    def optimisation_detector_placement_run(self, cmin, epsilon, N_dmax,  time_limit=1e5):
        """
               set up and solve the problem for minimising the overall cost of the network
               """
        t_0 = time.time()
        print("Start Optimisation")
        self._ensure_enough_detectors_on_node(epsilon = epsilon, N_dmax = N_dmax)
        self._flow_conservation()
        self._no_flow_out_sink_in_source()
        self._connectivity_constraint(cmin = cmin)
        self._objective_value()
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = create_sol_dict(self.prob)
        return sol_dict, self.prob

def split_sol_dict(sol_dict):
    """
       Split the solution dictionary into 4 dictionaries containing the fractional usage variables only and the binary
       variables only, lambda variables only and usage for keys variables
       Parameters
       ----------
       sol_dict : The solution dictionary containing solutions to the primary flow problem

       Returns : A dictionary with only the fractional detectors used, and a dictionary with only the binary values of
               whether the detector is on or off for cold and hot
       -------

       """
    flow_dict = {}
    binary_dict = {}
    y_dict = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "x":
            flow_dict[key] = sol_dict[key]
        elif key[0] == "d":
            binary_dict[key] = sol_dict[key]
        elif key[0] == "y":
            y_dict[key] = sol_dict[key]
    return flow_dict, binary_dict, y_dict


def plot_graphs(graph, delta_dict, save_extra = ""):
    graph = graph.to_undirected()
    pos = {}
    on_source_nodes = []
    off_source_nodes = []
    on_trusted_nodes = []
    off_trusted_nodes = []
    for key in delta_dict:
        current_node = int(key.split("_")[1])
        node_type = graph.nodes[current_node]["node_type"]

        on_off = delta_dict[key]
        if node_type == "S":
            if on_off:
                on_source_nodes.append(current_node)
            else:
                off_source_nodes.append(current_node)
        elif node_type == "B":
            if on_off:
                on_trusted_nodes.append(current_node)
            else:
                off_trusted_nodes.append(current_node)
    for node in graph.nodes:
        pos[node] = [graph.nodes[node]["xcoord"], graph.nodes[node]["ycoord"]]
    plt.figure()


    nx.draw_networkx_nodes(graph, pos, nodelist = on_source_nodes, node_color="r", label = "Detectors on Source Node")
    nx.draw_networkx_nodes(graph, pos, nodelist=off_source_nodes, node_color="b", label="No Detectors on Source Node")
    nx.draw_networkx_nodes(graph, pos, nodelist = on_trusted_nodes, node_color="g", label = "Detectors on Trusted Node")
    nx.draw_networkx_nodes(graph, pos, nodelist=off_trusted_nodes, node_color="k", label="No Detectors on Trusted Node")
    nx.draw_networkx_edges(graph, pos,edge_color="k")
    plt.axis("off")
    plt.legend(loc="best", fontsize="small")
    plt.savefig(f"plot_graph_{save_extra}.jpg")
    plt.show()


def make_key_dict_bidirectional(key_dict):
    """
    make the key dict bidirectional : if (source, target) in key_dict then (target, source) should be too
    """
    missing_entries = [(k[1],k[0]) for k in key_dict if (k[1],k[0]) not in key_dict]
    key_dict_copy = copy.deepcopy(key_dict)
    for idx in missing_entries:
        key_dict_copy[idx] = 0
    return key_dict_copy#

if __name__ == "__main__":
    key_dict, graphs = import_problem_detectors(node_file_path="1_node_data_different_sizes_no_users.csv",
                                                edge_file_path="1_capacities_different_no_users.csv", key_dict_path="1_key_dict_different_sizes_no_users.csv")
    optimal = {}
    for key in graphs.keys():
        prob = cplex.Cplex()
        key_dict_bidirect = make_key_dict_bidirectional(key_dict[key])
        optim = No_Switching_Detector_Optimisation(prob = prob, g = graphs[key], key_dict = key_dict_bidirect)
        sol_dict, prob = optim.optimisation_detector_placement_run(cmin = 100.0, epsilon = 0.001, N_dmax = 10,  time_limit=1e2)
        flow_dict, binary_dict, y_dict = split_sol_dict(sol_dict)
        plot_graphs(graphs[key], delta_dict= binary_dict)
        optimal[key]=prob.solution.get_objective_value()
    for key in graphs.keys():
        print(f"Graph {key}, number of detectors: "+ str(optimal[key]))
