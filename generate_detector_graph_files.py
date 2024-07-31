import generate_graph
import utils_graph
from enum import Enum, unique
import numpy as np
import trusted_nodes.vector as vector
import csv
import os

@unique
class Topology(Enum):
    """
    Enum to keep track of each topology - Bus = 0, Ring=1, Star=2, Mesh = 3, Hub & Spoke= 4
    """
    Bus = 0
    Ring = 1
    Star = 2
    Mesh = 3
    HubSpoke = 4


def generate_random_coordinates(n, box_size):
        xcoords = np.random.uniform(low=0.0, high=box_size, size=(n))
        ycoords = np.random.uniform(low=0.0, high=box_size, size=(n))
        return xcoords, ycoords

def generate_random_user_perturbation(n, no_users):
    # get an array of the appropriate number of nodes, bob nodes, and bob locations
    array_for_perturbation = np.concatenate((np.full(shape=no_users, fill_value=0),
                                             np.full(shape=n - no_users, fill_value=2)))
    # randomly permute this set of nodes to generate random graph
    node_types = np.random.permutation(array_for_perturbation)
    return node_types

def make_standard_labels():
    node_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14", "15"]
    for i in range(16, 500):
        node_names.append(str(i))
    return node_names

def generate_random_topology_graphs(topology,  no_nodes, no_users, box_size, dbswitch, no_of_conns_av = 3.5, nodes_for_mesh = 5):
    if Topology.Bus== topology:
        xcoords, ycoords = generate_random_coordinates(no_nodes,box_size )
        node_type = generate_random_user_perturbation(no_nodes, no_users)
        label = make_standard_labels()
        return generate_graph.BusNetwork(xcoords, ycoords, node_type, label, dbswitch)
    elif Topology.Ring ==topology:
        node_type = generate_random_user_perturbation(no_nodes, no_users)
        radius = np.random.uniform(low = box_size/10, high = box_size)
        label = make_standard_labels()
        return generate_graph.RingNetwork(radius, no_nodes, node_type, label, dbswitch)
    elif Topology.Star ==topology:
        xcoords, ycoords = generate_random_coordinates(no_nodes, box_size)
        node_type = generate_random_user_perturbation(no_nodes, no_users)
        label = make_standard_labels()
        central_node = 0
        magnitude = np.infty
        for i in range(len(node_type)):
            if node_type[i] == 2:
                current_vector = vector.Vector([xcoords[i]- box_size/2, ycoords[i] - box_size/2])
                magnitude_current = current_vector.magnitude()
                if magnitude_current < magnitude:
                    central_node = i
                    magnitude = magnitude_current
        return generate_graph.StarNetwork(xcoords, ycoords, node_type, central_node, label, dbswitch)
    elif Topology.Mesh == topology:
        label = make_standard_labels()
        graph_found = False
        while not graph_found:
            try:
                xcoords, ycoords = generate_random_coordinates(no_nodes, box_size)
                node_type = generate_random_user_perturbation(no_nodes, no_users)
                graph = generate_graph.MeshNetwork(xcoords, ycoords, node_type, no_of_conns_av, box_size, label, dbswitch)
                graph_found = True
            except ValueError:
                continue
        return graph
    elif Topology.HubSpoke == topology:
        label = make_standard_labels()

        graph_found = False
        while not graph_found:
            try:
                xcoords, ycoords = generate_random_coordinates(no_nodes, box_size)
                node_type = generate_random_user_perturbation(no_nodes, no_users)
                graph = generate_graph.HubSpokeNetwork(xcoords, ycoords, node_type, nodes_for_mesh, no_of_conns_av, box_size, label, dbswitch, mesh_in_centre = True)
                graph_found = True
            except ValueError:
                continue
        return graph


def get_capacity_edge(graph, dictionary, switched = False, db_switch = 1):
    edges_array = graph.get_edges(eprops=[graph.lengths_of_connections, graph.lengths_with_switch])
    edges_array = np.array(edges_array)
    capacities = {}
    for edge in edges_array:
        distance = edge[2]
        if switched == True:
            distance = distance + 5 * db_switch
        length = round(distance, 2)
        if length > 999:
            capacity = 0.0
        else:
            # from the look-up table
            capacity = dictionary["L" + str(length)]
        if capacity > 0.00000001:
            if (edge[0],edge[1]) in capacities.keys():
                capacities[edge[0], edge[1]].append((capacity, edge[2]))
            else:
                capacities[edge[0], edge[1]] = [(capacity, edge[2])]
    return capacities


def store_capacities(capacities, store_location, graph_id):
    for key in capacities.keys():
        dictionary_fieldnames = ["ID", "source_node", "target_node", "capacity", "distance"]
        values = {"ID": graph_id, "source_node": key[0], "target_node": key[1], "capacity" : capacities[key][0][0], "distance": capacities[key][0][1]}
        dictionary = [values]
        if os.path.isfile(store_location + '.csv'):
            with open(store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writerows(dictionary)
        else:
            with open(store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
                writer.writerows(dictionary)


def store_key_dict(graph, store_location, graph_id):
    vertices = graph.get_vertices()
    node_array = np.array(vertices)
    dictionary_fieldnames = ["ID", "source", "target", "key_value"]
    for node_1 in node_array:
        for node_2 in node_array:
            if node_1 < node_2 and graph.vertex_type[vertices[node_1]] == "S" and graph.vertex_type[vertices[node_2]] == "S":
                key_values = {"ID": graph_id, "source": node_1, "target": node_2, "key_value": 1}
                dictionary = [key_values]
                if os.path.isfile(store_location + '.csv'):
                    with open(store_location + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(store_location + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)


def store_position_data(graph, vertex_store_location, graph_id):
    dictionary_fieldnames = ["ID", "node", "node_type", "xcoord", "ycoord"]
    for node in range(len(graph.vertex_properties["x_coord"].a)):
        values = {"ID": graph_id, "node": node, "node_type": graph.vertex_type[node], "xcoord": graph.vertex_properties["x_coord"].a[node], "ycoord": graph.vertex_properties["y_coord"].a[node]}
        dictionary = [values]
        if os.path.isfile(vertex_store_location + '.csv'):
            with open(vertex_store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writerows(dictionary)
        else:
            with open(vertex_store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
                writer.writerows(dictionary)


if __name__ == "__main__":
    topology = Topology(3)
    no_nodes = 20
    no_users = 10
    box_size = 100
    db_switch = 1
    dictionary_bb84 = {}
    with open('rates_hotbob_bb84_20_eff.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_bb84["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')
    k = 0
    # for no_nodes in np.arange(10,100,10):
        # j = 0
    for no_users in np.arange(5,15,5):
        for i in range(20):
            graph= generate_random_topology_graphs(topology, no_nodes, no_users, box_size, db_switch, no_of_conns_av=10,
                                            nodes_for_mesh=5)
            capacities = get_capacity_edge(graph, dictionary_bb84, switched=False, db_switch=1)
            store_capacities(capacities, store_location = "1_capacities_different_no_users", graph_id = i + 20 * k)
            store_key_dict(graph, store_location = "1_key_dict_different_sizes_no_users", graph_id = i + 20 * k)
            store_position_data(graph, vertex_store_location = "1_node_data_different_sizes_no_users", graph_id = i + 20 * k)
            # j += 1
        k += 1
