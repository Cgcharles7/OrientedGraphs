import random
import json

def build_directed_oriented_graph(min_degree):
    from itertools import combinations
    
    def add_edges(graph, nodes):
        # Connect each pair of nodes in the subset with directed edges
        for i, j in combinations(nodes, 2):
            graph[i][j] = 1  # Forward edge only
    
    size = min_degree * (min_degree - 1) // 2 + min_degree  # Estimate the size of the graph
    graph = [[0] * size for _ in range(size)]
    
    current_node = 0
    while current_node < size:
        # Determine the number of new nodes
        num_new_nodes = min_degree - 1
        
        if current_node + num_new_nodes >= size:
            num_new_nodes = size - current_node - 1
        
        new_nodes = list(range(current_node + 1, current_node + 1 + num_new_nodes))
        
        # Connect the central node to new nodes
        for node in new_nodes:
            graph[current_node][node] = 1  # Oriented: only forward edge
        
        # Fully connect new nodes to each other with oriented (one-way) directed edges
        add_edges(graph, new_nodes)
        
        current_node += num_new_nodes + 1
    
    return graph


def build_directed_oriented_graph_with_backward_arcs(min_degree):
    from itertools import combinations
    
    def add_edges_with_backward(graph, nodes):
        # Connect each pair of nodes in the subset with directed edges
        for i, j in combinations(nodes, 2):
            graph[i][j] = 1  # Forward edge
            if (i < j):
                graph[j][i] = 1  # Backward edge
    
    size = min_degree * (min_degree - 1) // 2 + min_degree  # Estimate the size of the graph
    graph = [[0] * size for _ in range(size)]
    
    current_node = 0
    while current_node < size:
        # Determine the number of new nodes
        num_new_nodes = min_degree - 1
        
        if current_node + num_new_nodes >= size:
            num_new_nodes = size - current_node - 1
        
        new_nodes = list(range(current_node + 1, current_node + 1 + num_new_nodes))
        
        # Connect the central node to new nodes
        for node in new_nodes:
            graph[current_node][node] = 1
        
        # Fully connect new nodes to each other with directed edges and selective backward arcs
        add_edges_with_backward(graph, new_nodes)
        
        current_node += num_new_nodes + 1
    
    return graph

def generate_graph_data(num_graphs, min_degree_range, with_backward_arcs):
    graph_data = []
    for _ in range(num_graphs):
        min_degree = random.randint(*min_degree_range)
        if with_backward_arcs:
            graph = build_directed_oriented_graph_with_backward_arcs(min_degree)
        else:
            graph = build_directed_oriented_graph(min_degree)
        
        size = len(graph)
        num_sets = len([1 for i in range(size) if graph[i].count(1) >= min_degree])
        
        graph_data.append({
            'min_degree': min_degree,
            'size': size,
            'num_sets': num_sets,
            'graph': graph
        })
    
    return graph_data

def save_graph_data_to_json(filename, graph_data):
    with open(filename, 'w') as f:
        json.dump(graph_data, f, indent=2)

# Generate 100 graphs without backward arcs
graphs_no_backward = generate_graph_data(100, (3, 25), with_backward_arcs=False)
save_graph_data_to_json('graphs_no_backward.json', graphs_no_backward)

# Generate 100 graphs with backward arcs
graphs_with_backward = generate_graph_data(100, (3, 25), with_backward_arcs=True)
save_graph_data_to_json('graphs_with_backward.json', graphs_with_backward)
