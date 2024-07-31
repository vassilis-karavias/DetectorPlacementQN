import copy

import numpy as np
from copy import deepcopy
from random import randint, uniform
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import game_theory_model
from import_graph_data import import_problem_detectors

class Fitness_Function():


    def __init__(self, chromosome, origin_graph, ordering):
        self.chromosome = chromosome
        new_graph = deepcopy(origin_graph)
        C= {}
        j = 0
        for i in ordering:
            C[i] = chromosome.chromosome_list_form[j]
            j+= 1
        self.game = game_theory_model.Game(origin_graph, ordering, C = C)
        self.graph = new_graph


    def _calculate_fitness_function(self, n = 10):
        self.game.run_till_nash_equilibrium_or_n_rounds(n)
        return sum(self.game.C.values())

    def get_fitness_value(self, n = 10):
        return self._calculate_fitness_function(n)




class Chromosome():


    def __init__(self, dict_chromosome, graph):
        if len(dict_chromosome) != len(graph.nodes):
            print("Chromosomes not the right length")
            raise ValueError
        self.chromosome_list_form = dict_chromosome
        self.graph = graph

    def crossover(self, chromosome_2):
        dict_crossover_value = []
        dict_crossover_value_2 = []
        crossover_point = randint(2,len(self.chromosome_list_form) - 2)
        for i in range(len(self.chromosome_list_form)):
            if i < crossover_point:
                dict_crossover_value.append(self.chromosome_list_form[i])
                dict_crossover_value_2.append(chromosome_2.chromosome_list_form[i])
            else:
                dict_crossover_value.append(chromosome_2.chromosome_list_form[i])
                dict_crossover_value_2.append(self.chromosome_list_form[i])

        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph), Chromosome(dict_chromosome=dict_crossover_value_2, graph = self.graph)


    def crossover_2_point(self, chromosome_2):
        dict_crossover_value = []
        dict_crossover_value_2 = []
        crossover_point_1 = randint(1,len(self.chromosome_list_form) - 1)
        crossover_point_2 = randint(1, len(self.chromosome_list_form) - 2)
        print("Obtained Crossover Points")
        if crossover_point_1 == crossover_point_2:
            crossover_point_2 += 1
            print("Crossover Points Same: Correction made")
        i = 0
        if crossover_point_2 < crossover_point_1:
            print("crossover 2 smaller than 1. Starting swap")
            cp = crossover_point_2
            crossover_point_2 = crossover_point_1
            crossover_point_1 = cp
            print("Swap Made")
        for i in range(len(self.chromosome_list_form)):
            if i < crossover_point_1:
                dict_crossover_value.append(self.chromosome_list_form[i])
                dict_crossover_value_2.append(chromosome_2.chromosome_list_form[i])
            elif i < crossover_point_2:
                dict_crossover_value.append(chromosome_2.chromosome_list_form[i])
                dict_crossover_value_2.append(self.chromosome_list_form[i])
            else:
                dict_crossover_value.append(self.chromosome_list_form[i])
                dict_crossover_value_2.append(chromosome_2.chromosome_list_form[i])
        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph), Chromosome(dict_chromosome=dict_crossover_value_2, graph=self.graph)

    def crossover_3_point(self, chromosome_2):
        dict_crossover_value = []
        dict_crossover_value_2 = []
        crossover_point_1 = randint(1, len(self.chromosome_list_form) - 1)
        crossover_point_2 = randint(1, len(self.chromosome_list_form) - 2)
        crossover_point_3 = randint(1, len(self.chromosome_list_form) - 3)
        if crossover_point_1 == crossover_point_2:
            crossover_point_2 += 1
        if crossover_point_1 == crossover_point_3:
            crossover_point_3 += 2
        if crossover_point_2 == crossover_point_3:
            crossover_point_3 += 1
        i = 0

        if crossover_point_2 < crossover_point_1:
            cp = crossover_point_2
            crossover_point_2 = crossover_point_1
            crossover_point_1 = cp
        if crossover_point_3 < crossover_point_2:
            cp = crossover_point_2
            crossover_point_2 = crossover_point_3
            crossover_point_3 = cp
        if crossover_point_2 < crossover_point_1:
            cp = crossover_point_2
            crossover_point_2 = crossover_point_1
            crossover_point_1 = cp
        for i in range(len(self.chromosome_list_form)):
            if i < crossover_point_1:
                dict_crossover_value.append(self.chromosome_list_form[i])
                dict_crossover_value_2.append(chromosome_2.chromosome_list_form[i])
            elif i < crossover_point_2:
                dict_crossover_value.append(chromosome_2.chromosome_list_form[i])
                dict_crossover_value_2.append(self.chromosome_list_form[i])
            elif i < crossover_point_3:
                dict_crossover_value.append(self.chromosome_list_form[i])
                dict_crossover_value_2.append(chromosome_2.chromosome_list_form[i])
            else:
                dict_crossover_value.append(chromosome_2.chromosome_list_form[i])
                dict_crossover_value_2.append(self.chromosome_list_form[i])
        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph), Chromosome(dict_chromosome=dict_crossover_value_2, graph=self.graph)

    def crossover_half_uniform(self, chromosome_2):
        ## might need a different crossover method?
        dict_crossover_value = []
        for i in range(len(self.chromosome_list_form)):
            if self.chromosome_list_form[i] == chromosome_2.chromosome_list_form[i]:
                dict_crossover_value.append(self.chromosome_list_form[i])
            else:
                dict_crossover_value.append(randint(0, 1))
        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph)


    def crossover_three_parents(self, chromosome_2, chromosome_3):
        ## might need a different crossover method?
        dict_crossover_value = []
        for i in range(len(self.chromosome_list_form)):
            if self.chromosome_list_form[i] == chromosome_2.chromosome_list_form[i]:
                dict_crossover_value.append(self.chromosome_list_form[i])
            else:
                dict_crossover_value.append(chromosome_3.chromosome_list_form[i])
        return Chromosome(dict_chromosome=dict_crossover_value, graph = self.graph)



    def mutation(self, prob_mutation):
        if prob_mutation > 1:
            print("prob_mutation > 1")
            raise ValueError
        new_chromosome_dict = []
        for i in range(len(self.chromosome_list_form)):
            print("Getting Random Value")
            random_value = np.random.uniform()
            if random_value <= prob_mutation:
                print("Mutation Occuring")
                new_chromosome_dict.append((self.chromosome_list_form[i]+ 1)% 2)
            else:
                print("No Mutation")
                new_chromosome_dict.append(self.chromosome_list_form[i])
        print("Mutation Completed")
        return Chromosome(dict_chromosome= new_chromosome_dict, graph = self.graph)



def takeSecond(elem):
    return elem[1]

class Population():

    def __init__(self, list_chromosomes, graph, ordering):
        self.list_chromosomes = list_chromosomes
        self.graph = graph
        self.ordering = ordering


    def selection(self, number_parents_in_next_population):
        if number_parents_in_next_population > len(self.list_chromosomes):
            raise ValueError
        chromosomes_to_keep = []
        current_needed_fitness_value = 1000000000 ### we want to exclude only solutions that are not feasible
        i =0
        for chromosome in self.list_chromosomes:
            fitness = Fitness_Function(chromosome= chromosome, origin_graph= self.graph, ordering=self.ordering)
            fitness_value = fitness.get_fitness_value()
            if fitness_value < current_needed_fitness_value:
                if len(chromosomes_to_keep) < number_parents_in_next_population:
                    chromosomes_to_keep.append((chromosome, fitness_value))
                    chromosomes_to_keep.sort(key=takeSecond)
                else:
                    chromosomes_to_keep = chromosomes_to_keep[:-1] + [(chromosome, fitness_value)]
                    chromosomes_to_keep.sort(key=takeSecond)
                    current_needed_fitness_value = chromosomes_to_keep[-1][1]
            # print("Finished chromosome " + str(i))
            # i += 1
        chromosomes = []
        fittest_chromosome = chromosomes_to_keep[0][0]
        for chromosome, fitness_value in chromosomes_to_keep:
            chromosomes.append(chromosome)
        return chromosomes, fittest_chromosome


    def selection_fitness_proportionate_selection(self, number_parents_in_next_population):
        if number_parents_in_next_population > len(self.list_chromosomes):
            raise ValueError
        chromosomes_to_keep = []
        total_fitness = 0.0
        i=0
        for chromosome in self.list_chromosomes:
            fitness = Fitness_Function(chromosome=chromosome, origin_graph=self.graph, ordering=self.ordering)
            fitness_value = fitness.get_fitness_value()
            if i == 0:
                fitness_value_for_largest = fitness_value
            chromosomes_to_keep.append((chromosome,fitness_value))
            if fitness_value != np.infty:
                total_fitness += fitness_value
            # print("Finished chromosome " + str(i))
            i += 1
        chromosomes = []
        ### chromomsome 1 is always the one for reducing number by one:
        chromosome = self.list_chromosomes[0]

        ### This is for the chromosome reduction selection
        # new_chromosome = chromosome.get_next_best_with_less_values(key_dict= self.key_dict,Lambda = self.Lambda, f_switch = self.f_switch, C_det= self.C_det,
        #                                                   C_source= self.C_source, c_on= self.c_on, cmin= self.cmin, fitness_value_this_chromosome = fitness_value_for_largest)
        # if new_chromosome == None:
        #     chromosomes = []
        #     return chromosomes, None
        # else:
        #     chromosomes.append(new_chromosome)
        # use stochastic acceptance
        fittest_chromosome = min(chromosomes_to_keep, key = lambda t: t[1])
        while len(chromosomes) < number_parents_in_next_population:
            total_reduce_fitness = 0.0
            for chromosome, fitness_value in chromosomes_to_keep:
                prob = uniform(0,1)
                if fitness_value != np.infty:
                    if prob < fitness_value/total_fitness:
                        chromosomes.append(chromosome)
                        chromosomes_to_keep.remove((chromosome, fitness_value))
                        total_reduce_fitness += fitness_value
            total_fitness = total_fitness - total_reduce_fitness
            if total_fitness < 0.0001:
                break
        return chromosomes, fittest_chromosome


    def generate_next_population(self, number_parents_in_next_population, next_population_size, p_cross, prob_mutation):
        ## p_cross -> probability of using crossing, prob_mutation is the probability that a gene undergoes a mutation
        parent_chromosomes, fittest_chromosome = self.selection(number_parents_in_next_population=number_parents_in_next_population)
        print("Obtained Parent Chromosomes")
        if fittest_chromosome == None:
            return None, None
        next_generation = []
        p_mut = 1-p_cross
        while len(next_generation) < next_population_size:
            prob = uniform(0, 1)
            if prob < p_mut:
                print("Mutation Started")
                parent = randint(0,len(parent_chromosomes)-1)
                print("Mutation Function Entered")
                child_chromosome = parent_chromosomes[parent].mutation(prob_mutation)
                next_generation.append(child_chromosome)
                print("Mutated")
            else:
                print("Crossover Started")
                parent_1 = randint(0, len(parent_chromosomes) - 1)
                parent_2 = randint(0, len(parent_chromosomes) - 2)
                if parent_2 == parent_1 and parent_2 != len(parent_chromosomes) - 1:
                    parent_2 += 1
                elif parent_2 == parent_1:
                    parent_2 -= 1
                child_chromosome_1= parent_chromosomes[parent_1].crossover_half_uniform(parent_chromosomes[parent_2])
                next_generation.extend([child_chromosome_1])
                print("Crossover Made")
        print("Next Generation Population Made")
        return Population(list_chromosomes = next_generation, graph = self.graph, ordering = self.ordering), fittest_chromosome

    def select_best_member_of_population(self):
        return self.selection(number_parents_in_next_population=1)


class Heuristic_Genetic():

    def __init__(self, graph, ordering):
        self.graph = graph
        self.ordering = ordering


    def generate_initial_population(self, population_size):
        #### full node on is a member of all initial populations and used to check existence of solution Returns None
        #### if no solution exists

        chromosomes = []

        full_gene = []
        for i in range(len(self.graph.nodes)):
            full_gene.append(0)
        chromosome_1 = Chromosome(dict_chromosome = full_gene, graph = self.graph)

        fitness = Fitness_Function(chromosome = chromosome_1, origin_graph = self.graph, ordering=self.ordering)
        fitness_value = fitness.get_fitness_value()
        if fitness_value == np.infty:
            return None
        chromosomes.append(chromosome_1)
        while len(chromosomes) < population_size:
            chromosome_dict = np.random.randint(low = 0, high = 2, size = len(self.graph.nodes))
            chromosome = Chromosome(dict_chromosome = chromosome_dict, graph = self.graph)
            chromosomes.append(chromosome)
        return Population(list_chromosomes = chromosomes, graph = self.graph, ordering = self.ordering)



    def single_step(self, current_population, number_parents_in_next_population, next_population_size, p_cross, prob_mutation):
        return current_population.generate_next_population(number_parents_in_next_population, next_population_size, p_cross, prob_mutation)


    def full_recursion(self, number_parents_in_next_population, next_population_size, p_cross, prob_mutation, number_steps):
        if number_parents_in_next_population > next_population_size:
            print("Number of parents for next generation cannot be bigger than size of parent population")
            raise ValueError
        if p_cross > 1:
            print("CrossOver probability cannot exceed 1")
            raise ValueError
        if prob_mutation >1:
            print("Probability of gene mutation cannot exceed 1")
            raise ValueError
        initial_population = self.generate_initial_population(population_size=next_population_size)
        print("Initial Population Generated")
        current_population = initial_population
        if initial_population == None:
            print("No solution exists")
            raise ValueError
        else:
            prev_best = 100000000
            fittest_chromosomes = []
            for i in range(number_steps):
                print("Starting Step " + str(i))
                current_population_curr, fittest_chromosome_curr = self.single_step(current_population = current_population, number_parents_in_next_population = number_parents_in_next_population, next_population_size = next_population_size, p_cross = p_cross, prob_mutation = prob_mutation)
                print("Ending Step " + str(i))
                if current_population_curr == None:
                    return min(fittest_chromosomes, key = lambda t: t[1])
                current_population = current_population_curr
                fittest_chromosomes.append(fittest_chromosome_curr)
                # print("Current Difference: " + str(prev_best - fittest_chromosome[1]))
                # if prev_best - fittest_chromosome[1] < 0.05 and prev_best - fittest_chromosome[1] >= 0.0:
                #     break
                # if prev_best - fittest_chromosome[1] > 0:
                #     prev_best = fittest_chromosome[1]
            print("Finished Steps. Getting best chromosome")
            best_chromosome = current_population.select_best_member_of_population()
            fitness = Fitness_Function(chromosome= best_chromosome[0][0], origin_graph= self.graph, ordering=self.ordering)
            fitness_value = fitness.get_fitness_value()
            return best_chromosome, fitness_value


def plot_centrality_models_genetic_models(data_storage_optimal_location, data_storage_centrality_measures, data_storage_genetic, data_storage_game_theory):


    objective_values_optimal = {}
    if data_storage_optimal_location != None:
        plot_information = pd.read_csv(data_storage_optimal_location + ".csv")
        for index, row in plot_information.iterrows():
            objective_values_optimal[row["Graph key"]] =  row["objective_value"]

    objective_values_models = {}
    if data_storage_centrality_measures != None:
            plot_information = pd.read_csv(data_storage_centrality_measures + ".csv")
            for index, row in plot_information.iterrows():

                if row["number_nodes"] not in objective_values_models.keys():
                    objective_values_models[row["number_nodes"]]  = {row["Graph key"] : row["objective_value"]}#
                else:
                    objective_values_models[row["number_nodes"]][row["Graph key"]] = row["objective_value"]
    objective_values_genetic_algorithm_models = {}
    if data_storage_genetic != None:
        plot_information = pd.read_csv(data_storage_genetic + ".csv")
        for index, row in plot_information.iterrows():

            if row["number_nodes"] not in objective_values_genetic_algorithm_models.keys():
                objective_values_genetic_algorithm_models[row["number_nodes"]] = {row["Graph key"]: row["objective_value"]}  #
            else:
                objective_values_genetic_algorithm_models[row["number_nodes"]][row["Graph key"]] = row["objective_value"]
    objective_values_game_theory = {}
    if data_storage_game_theory != None:
        plot_information = pd.read_csv(data_storage_game_theory + ".csv")
        for index, row in plot_information.iterrows():

            if row["number_nodes"] not in objective_values_game_theory.keys():
                objective_values_game_theory[row["number_nodes"]] = {
                    row["Graph key"]: row["objective_value"]}  #
            else:
                objective_values_game_theory[row["number_nodes"]][row["Graph key"]] = row[
                    "objective_value"]
    objective_values = {}
    for number_nodes in objective_values_models.keys():
        for graph_key in objective_values_models[number_nodes].keys():
            if graph_key in objective_values_optimal.keys():
                if number_nodes not in objective_values.keys():
                    objective_values[number_nodes] = [(objective_values_models[number_nodes][graph_key] - objective_values_optimal[graph_key])/  objective_values_optimal[graph_key]]
                else:
                    objective_values[number_nodes].append((objective_values_models[number_nodes][graph_key] - objective_values_optimal[graph_key])/  objective_values_optimal[graph_key])
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
    plt.errorbar(x, mean_differences, yerr=std_differences, label = f"Degree Centrality", marker = "o")
    objective_values_genetic = {}
    for number_nodes in objective_values_genetic_algorithm_models.keys():
        for graph_key in objective_values_genetic_algorithm_models[number_nodes].keys():
            if graph_key in objective_values_optimal.keys():
                if number_nodes not in objective_values_genetic.keys():
                    objective_values_genetic[number_nodes] = [
                        (objective_values_genetic_algorithm_models[number_nodes][graph_key] - objective_values_optimal[graph_key]) /
                        objective_values_optimal[graph_key]]
                else:
                    objective_values_genetic[number_nodes].append(
                        (objective_values_genetic_algorithm_models[number_nodes][graph_key] - objective_values_optimal[graph_key]) /
                        objective_values_optimal[graph_key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values_genetic.keys():
        mean_objectives[key] = np.mean(objective_values_genetic[key])
        std_objectives[key] = np.std(objective_values_genetic[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, label=f"Genetic Model", marker="o")
    objective_values_game = {}
    for number_nodes in objective_values_game_theory.keys():
        for graph_key in objective_values_game_theory[number_nodes].keys():
            if graph_key in objective_values_optimal.keys():
                if number_nodes not in objective_values_game.keys():
                    objective_values_game[number_nodes] = [
                        (objective_values_game_theory[number_nodes][graph_key] - objective_values_optimal[graph_key]) /
                        objective_values_optimal[graph_key]]
                else:
                    objective_values_game[number_nodes].append(
                        (objective_values_game_theory[number_nodes][graph_key] - objective_values_optimal[graph_key]) /
                        objective_values_optimal[graph_key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values_game.keys():
        mean_objectives[key] = np.mean(objective_values_game[key])
        std_objectives[key] = np.std(objective_values_game[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, label=f"Game Theory with random C and O", marker="o")
    plt.xlabel("Number of Nodes", fontsize=10)
    plt.ylabel("Percentage Difference from optimal", fontsize=10)
    plt.legend()
    plt.savefig("genetic_vs_centrality_vs_game_comparison.png")
    plt.show()

def plot_genetic_algorithm(data_storage_optimal_location, data_storage_genetics, genetics_list):


    objective_values_optimal = {}
    if data_storage_optimal_location != None:
        plot_information = pd.read_csv(data_storage_optimal_location + ".csv")
        for index, row in plot_information.iterrows():
            objective_values_optimal[row["Graph key"]] =  row["objective_value"]
    i = 0
    for data_storage_genetic in data_storage_genetics:
        objective_values_genetic_algorithm_models = {}
        if data_storage_genetic != None:
            plot_information = pd.read_csv(data_storage_genetic + ".csv")
            for index, row in plot_information.iterrows():

                if row["number_nodes"] not in objective_values_genetic_algorithm_models.keys():
                    objective_values_genetic_algorithm_models[row["number_nodes"]] = {row["Graph key"]: row["objective_value"]}  #
                else:
                    objective_values_genetic_algorithm_models[row["number_nodes"]][row["Graph key"]] = row["objective_value"]
        objective_values_genetic = {}
        for number_nodes in objective_values_genetic_algorithm_models.keys():
            for graph_key in objective_values_genetic_algorithm_models[number_nodes].keys():
                if graph_key in objective_values_optimal.keys():
                    if number_nodes not in objective_values_genetic.keys():
                        objective_values_genetic[number_nodes] = [
                            (objective_values_genetic_algorithm_models[number_nodes][graph_key] - objective_values_optimal[graph_key]) /
                            objective_values_optimal[graph_key]]
                    else:
                        objective_values_genetic[number_nodes].append(
                            (objective_values_genetic_algorithm_models[number_nodes][graph_key] - objective_values_optimal[graph_key]) /
                            objective_values_optimal[graph_key])
        mean_objectives = {}
        std_objectives = {}
        for key in objective_values_genetic.keys():
            mean_objectives[key] = np.mean(objective_values_genetic[key])
            std_objectives[key] = np.std(objective_values_genetic[key])
        mean_differences = []
        std_differences = []
        # topologies
        x = []
        for key in mean_objectives.keys():
            mean_differences.append(mean_objectives[key])
            std_differences.append(std_objectives[key])
            x.append(key)
        plt.errorbar(x, mean_differences, yerr=std_differences, label=genetics_list[i], marker="o")
        i += 1
    plt.xlabel("Number of Nodes", fontsize=10)
    plt.ylabel("Percentage Difference from optimal", fontsize=10)
    plt.legend()
    plt.savefig("genetic_comparison.png")
    plt.show()

if __name__ == "__main__":
    # plot_centrality_models_genetic_models(data_storage_optimal_location = "3_optimal_solution_storage", data_storage_centrality_measures = "3_centrality_method_storage_degree",
    #                                       data_storage_genetic = "3_genetic_solution_fixed", data_storage_game_theory = "3_game_theory_storage_random_C_O")
    plot_genetic_algorithm(data_storage_optimal_location = "3_optimal_solution_storage",
                           data_storage_genetics =  ["3_genetic_solution_fixed_2_steps","3_genetic_solution_fixed_5_steps", "3_genetic_solution_fixed_10_steps",
                                                     "3_genetic_solution_fixed","3_genetic_solution_fixed_5_steps_20_individuals",
                                                    "3_genetic_solution_fixed_5_steps_50_individuals"], genetics_list = ["n=2, m=100", "n=5, m=100",
                                                                                                                         "n=10, m=100", "n=100, m=100",
                                                                                                                         "n=5, m=20", "n=5, m=50"])
    # key_dict, graphs = import_problem_detectors(node_file_path="3_node_data_different_sizes_no_users.csv",
    #                                             edge_file_path="3_capacities_different_no_users.csv", key_dict_path="3_key_dict_different_sizes_no_users.csv")
    # data_storage_location_keep_each_loop_simple_model = "3_genetic_solution_fixed_5_steps_20_individuals"
    # if os.path.isfile(data_storage_location_keep_each_loop_simple_model + '.csv'):
    #     plot_information = pd.read_csv(data_storage_location_keep_each_loop_simple_model + '.csv')
    #     last_row_explored = plot_information.iloc[[-1]]
    #     current_key = last_row_explored["Graph key"].iloc[0]
    # else:
    #     current_key = None
    #     dictionary_fieldnames = ["Graph key", "number_nodes", "objective_value"]
    #     with open(data_storage_location_keep_each_loop_simple_model + '.csv', mode='a') as csv_file:
    #         writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
    #         writer.writeheader()
    # for key in graphs.keys():
    #     if current_key != None and current_key != key:
    #         continue
    #     elif current_key != None:
    #         current_key = None
    #         continue
    #     ordering = game_theory_model.true_centrality_ordering(graphs[key])
    #     heuristic = Heuristic_Genetic(graph = graphs[key], ordering= ordering)
    #     try:
    #         chromosome, fitness_value = heuristic.full_recursion(number_parents_in_next_population = 10, next_population_size = 100, p_cross= 0.7, prob_mutation= 0.1, number_steps =100)
    #         print("Best Value: " + str(fitness_value))
    #
    #         if data_storage_location_keep_each_loop_simple_model != None:
    #             dictionary = [
    #                 {"Graph key": key, "number_nodes": len(graphs[key].nodes), "objective_value": fitness_value}]
    #             dictionary_fieldnames = ["Graph key", "number_nodes", "objective_value"]
    #             if os.path.isfile(data_storage_location_keep_each_loop_simple_model + '.csv'):
    #                 with open(data_storage_location_keep_each_loop_simple_model + '.csv',
    #                           mode='a') as csv_file:
    #                     writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
    #                     writer.writerows(dictionary)
    #             else:
    #                 with open(data_storage_location_keep_each_loop_simple_model + '.csv',
    #                           mode='a') as csv_file:
    #                     writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
    #                     writer.writeheader()
    #                     writer.writerows(dictionary)
    #
    #     except:
    #         print("No solution")
    #         continue
