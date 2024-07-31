# DetectorPlacementQN
Optimise the detector node placement in QNs

## Requirements
numpy: 1.20.1+  
graph-tool: 2.37+  
pandas: 1.5.2+  
scipy: 1.7.3+  
cplex: V20.1  
matplotlib: 3.6.2+  
networkx: 2.8.8+  

## Preprocessing
To generate the needed csv files, the python file *generate_detector_graph_files.py* can be used. To specify the topology of the graphs, the Enum Topology can be selected, the key rate dictionary can be imported using:  
*dictionary_bb84 = {}  
    with open('rates_hotbob_bb84_20_eff.csv', mode='r') as csv_file:  
        csv_reader = csv.DictReader(csv_file)  
        line_count = 0  
        for row in csv_reader:  
            if line_count == 0:  
                line_count += 1  
            dictionary_bb84["L" + str(round(float(row["L"]), 2))] = float(row['rate'])  
            line_count += 1  
        print(f'Processed {line_count} lines.')*  
Then the graphs can be generated using the methods:  
*graph= generate_random_topology_graphs(topology, no_nodes, no_users, box_size, db_switch, no_of_conns_av, nodes_for_mesh)*  
*topology* is the topology of the graph desired, *no_nodes* is the number of nodes in the graphs, *no_users* is the number of user nodes in the TN network, *box_size* is the km distance of the box in which the network is contained, *db_switch* is the dB loss of the switches - if no switches are used set this to 0, *no_of_conns_av* is only needed if topology is *Mesh* or *HubSpoke* and this specifies the average number of edges per node. Finally, *nodes_for_mesh* is only needed if topology is *HubSpoke* and specifies how many of the nodes are part of the mesh topology.  
Then the capacities can be obtained by  
*capacities = get_capacity_edge(graph, dictionary_bb84, switched, db_switch)*  
The *switched* parameter is a boolean denoting whether the graph is using switching or not, and *db_switch* is the dB loss of the switches. Finally, you can store the relevant csv files using:  
*store_capacities(capacities, store_location, graph_id)  
 store_key_dict(graph, store_location, graph_id)  
 store_position_data(graph, vertex_store_location, graph_id)*  
*store_location* is the string of where to store the files (no *.csv* ending is needed). *graph_id* is the ID value of the current graph and allows multiple graphs to be stored in a single .csv file. The first function stores the capacities of each edge in a .csv file with columns *["ID", "source_node", "target_node", "capacity", "distance"]* where *distance* is the distance of the connection in km. The second function stores the desired connectivity of parties in a csv file with columns *["ID", "source", "target", "key_value"]*. *key_value* is how many key paths are needed between the source and target, currently this is only set up for *key_value = 1*. Finally, the third function stores the node position data in a csv file with columns * ["ID", "node", "node_type", "xcoord", "ycoord"]*.  
## Optimisation and Heuristics
To import the csv data into usable format, the   
*key_dicts, graphs = import_graph_data.import_problem_detectors(node_file_path, edge_file_path, key_dict_path)*  
function can be used. *node_file_path* is the location of the node position data saved in the preprocessing, *edge_file_path* is the location of the edge capacity data and *key_dict_path* is the location of the key dictionary file. This outputs a dictionary of key dictionaries indexed by graph ID: *key_dicts = {ID: key_dict}* and a dictionary of graphs indexed by graph ID: *graphs = {ID: graph}*. To perform the optimisation, first you need to make the *key_dict* bidirectional:  
*key_dict_bidirect = optimisation_detector_no_switching.make_key_dict_bidirectional(key_dict[key])*  
And then you can set up and run the problem using  
*prob = cplex.Cplex()  
optim = optimisation_detector_no_switching.No_Switching_Detector_Optimisation(prob, g = graphs[key], key_dict = key_dict_bidirect)  
sol_dict, prob = optim.optimisation_detector_placement_run(cmin, epsilon, N_dmax,  time_limit)*  
*cmin* is the minimum capacity of each network connection in *key_dict_bidirect* in bits/s. *epsilon* is a small positive number to ensure the ceiling function is linearised appropriately (epsilon = 0.001 or less should be used), *N_dmax* is the maximum number of detectors on a single node and *time_limit* is the maximum amount of time to run the model. To get the objective value use: *prob.solution.get_objective_value()*  
The centrality model can be investigated using the class *Centrality_Model* in *centrality_model.py*. First, set up the class:  
*cm = Centrality_Model(graphs[key])*  
and then use the appropriate method dependent on the centrality desired:
*cm.degree_centrality()  
cm.betweenness_centrality_network_x()  
cm.eccentricity_centrality()  
cm.closeness_centrality_networkx()  
cm.integrated_path_centrality()  
cm.stress_centrality()  
cm.katz_centrality(alpha)  
cm.bonachich_eigenvector_centrality()  
cm.epsilon_betweenness_centrality(epsilon)*  
For the definitions of *alpha* and *epsilon* here, refer to the definitions of the Katz centrality metric and the epsilon-betweenness centrality metric. The detector locations can then be extracted by  
*D = cm.get_detector_set()*  
The number of detector sites is *len(D)*.  
To evaluate the game theory model, the classes in *game_theory_model.py* can be used. First, get a list of ordering of the nodes of the graph. For example, a random ordering can be obtained by *ordering = list(graphs[key].nodes)*, but a more sophisticated ordering based on the centrality order can be obtained by *ordering = game_theory_model.true_centrality_ordering(graphs[key])*. Then set up a game class  
*game = Game(graphs[key], ordering = ordering)*  
and run the game for n rounds:  
*game.run_till_nash_equilibrium_or_n_rounds(n)*  
To extract the total number of detector sites: *sum(game.C.values())* where *C* is a dict with *{node: on/off}* and is the strategy profile of the game. Note that in general, *C* is initialised randomly.  
To use the genetic algorithm model, the classes in *genetic_model.py* can be used. Again, a list of ordering of the nodes of the graph is needed. Once this is obtained, you can initialise the genetic algorithm class using:  
*heuristic = genetic_model.Heuristic_Genetic(graph = graphs[key], ordering= ordering)*  
To run the genetic model recursion, use:  
*chromosome, fitness_value = heuristic.full_recursion(number_parents_in_next_population, next_population_size, p_cross, prob_mutation, number_steps)*  
The *number_parents_in_next_population* is the number of parents used in to generate the next population, *next_population_size* is the size of each population, *p_cross* is the probability of a crossover occuring in to generate the child, while *1-p_cross* is the probability that a mutation is used instead. *prob_mutation* is the probability that an individual genome of the chromosome is mutated given the child is generated using the mutation method and *number_steps* is the number of generations to use. The number of detectors of the solution is given by *fitness_value* and *chromosome* gives the list of whether the node is on or off. Note that if there is no solution the recursion will throw an exception. 




