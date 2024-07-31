import cplex
import time
import os
import csv
import networkx as nx
import optimisation_detector_no_switching


class Switching_Detector_Optimisation(optimisation_detector_no_switching.Detector_Optimisation):

    def _ensure_enough_detectors_on_node(self, f_switch, N_dmax):
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        deltas = []
        for i in self.g.nodes:
            deltas.append(f"delta_{i}")
        self.prob.variables.add(names = deltas, types = [self.prob.variables.type.binary] * len(deltas))
        for i in self.g.nodes():
            ind = []
            val = []
            for j in self.g.adj[i]:
                capacity = int(self.g.edges[[i, j]]["capacity"])
                if capacity == 0:
                    ind_zero = []
                    val_zero = []
                    for k in self.key_dict:
                        ind_zero.extend([f"x{j}_{i}_k{k[0]}_{k[1]}"])
                        val_zero.extend([1])
                    constraint = [cplex.SparsePair(ind=ind_zero, val=val_zero)]
                    self.prob.linear_constraints.add(lin_expr=constraint, senses=["L"],
                                                     rhs=[0.0])
                else:
                    for k in self.key_dict:
                        ind.extend([f"x{j}_{i}_k{k[0]}_{k[1]}"])
                        val_zero.extend([1/capacity])
                    ind.extend([f"delta_{i}"])
                    val.extend([-(1-f_switch) * N_dmax])
                    constraint = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=constraint, senses=["L"],
                                                     rhs=[0.0])

    def optimisation_detector_placement_run(self, cmin, f_switch, N_dmax,  time_limit=1e5):
        """
               set up and solve the problem for minimising the overall cost of the network
               """
        t_0 = time.time()
        print("Start Optimisation")
        self._ensure_enough_detectors_on_node(f_switch = f_switch, N_dmax = N_dmax)
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
        sol_dict = optimisation_detector_no_switching.create_sol_dict(self.prob)
        return sol_dict, self.prob
