class Graph:
    def __init__(self, graph_data, is_matrix=False):
        self.graph = graph_data
        self.is_matrix = is_matrix

    def analyze_path(self, path):
        backward_arcs = self.detect_backward_arcs()
        path_nodes = set(path)

        for node in path:
            if len(self.graph[node]) >= 2 * len(path_nodes):
                return "Node with degree double"

        if backward_arcs:
            return "Backward arcs found"

        return "Path is valid"
    
    def bfs_distance(self, start_node, end_node):
        from collections import deque
        queue = deque([(start_node, 0)])  # (node, distance)
        visited = set()

        while queue:
            current_node, current_distance = queue.popleft()

            if current_node == end_node:
                return current_distance

            if current_node not in visited:
                visited.add(current_node)

                neighbors = self.graph[current_node] if not self.is_matrix else [i for i, is_neighbor in enumerate(self.graph[current_node]) if is_neighbor]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, current_distance + 1))

        return float('inf')  # Return infinity if there's no path between start_node and end_node

    def build_decreasing_path(self, start_node):
        path = [start_node]
        visited = set([start_node])

        self.dfs(start_node, path, visited)
        return path

    def create_path(self, start_node, max_depth):
        import random  # Ensure random is imported if it's used here.
        path = [start_node]
        visited = set([start_node])

        for _ in range(max_depth):
            current_node = path[-1]
            neighbors = self.first_degree_neighbors(current_node)

            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if not unvisited_neighbors:
                break

            next_node = random.choice(unvisited_neighbors)
            path.append(next_node)
            visited.add(next_node)

        return path

    def decreasing_sequence_property(self, x):
        N_plus_x = self.first_degree_neighbors(x)
        N_plus_plus_x = self.second_degree_neighbors(x)

        has_decreasing_property = len(N_plus_x) > len(N_plus_plus_x)

        if has_decreasing_property:
            interior_degrees = {y: len(self.interior_neighbors(y)) for y in N_plus_x}
            exterior_degrees = {y: len(self.exterior_neighbors(y)) for y in N_plus_x}
            return True, interior_degrees, exterior_degrees
        else:
            return False, None, None
        
    def deg(self, node):
        return len(self.first_degree_neighbors(node));

    def detect_backward_arcs(self):
        backward_arcs = []
        visited = {node: 'white' for node in self.graph}

        for node in self.graph:
            if visited[node] == 'white':
                self.dfsbw(node, visited, backward_arcs)

        return backward_arcs

    def dfs(self, current_node, path, visited):
        if len(path) > 1 and not self.has_decreasing_sequence_property(current_node):
            return False
        
        for neighbor in self.graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if not self.dfs(neighbor, path, visited):
                    return False
                path.pop()
                visited.remove(neighbor)
        return True
		
    def dfsbw(self, node, visited, backward_arcs):
        visited[node] = 'gray'  # Mark the node as being visited

        for neighbor in self.graph.get(node, []):
            if visited.get(neighbor, 'white') == 'gray':
                backward_arcs.append((node, neighbor))
            elif visited.get(neighbor, 'white') == 'white':
                self.dfsbw(neighbor, visited, backward_arcs)

        visited[node] = 'black'  # Mark the node as fully visited
    
    def dispAdjMatrix(self):
        for i in range(len(self.graph)):
            print(self.graph[i])
			
    def distance(self, node1, node2):
        return self.bfs_distance(node1, node2)

    def distance_k_decreasing_sequence_property(self, v0, k):
        distances = self.bfs_distance(v0)
        for node, dist in enumerate(distances):
            if 1 <= dist <= k and not self.has_decreasing_sequence_property(node):
                return False
        return True
    
    def exterior_neighbors(self, y):
        N_plus_y = self.first_degree_neighbors(y)
        return [neighbor for neighbor in N_plus_y if neighbor in self.second_degree_neighbors(y) and neighbor not in N_plus_y]

    def ext_w_x(self, node, w):
        N_plus_w = self.first_degree_neighbors(w)
        ext_neighbors = []
        for x in N_plus_w:
            out_neighbors_x = self.first_degree_neighbors(x)
            ext_neighbors.extend([i for i in out_neighbors_x if i not in N_plus_w])
        return list(set(ext_neighbors))

    def find_interior_exterior_seymour_diamonds(self):
        num_nodes = len(self.graph)
        diamonds = []

        for w in range(num_nodes):
            N_plus_w = self.first_degree_neighbors(w)
            for x in N_plus_w:
                int_neighbors = self.int_w_x(x, w)
                ext_neighbors = self.ext_w_x(x, w)

                for ui in int_neighbors:
                    for ue in ext_neighbors:
                        if any(self.graph[ui][y] and self.graph[ue][y] for y in range(num_nodes)):
                            diamonds.append((w, x, ui, ue, y))

        return diamonds

    def find_min_degree_node(self):
        min_degree = float('inf')
        min_node = None

        if isinstance(self.graph, list) and isinstance(self.graph[0], list):
            print("here")
            num_nodes = len(self.graph)
            for i in range(num_nodes):                    
                degree = sum(self.graph[i])
                if degree < min_degree:
                    min_degree = degree
                    min_node = i

        elif isinstance(self.graph, dict):
            print("here2")
            for node in self.graph:
                degree = len(self.graph[node])
                if degree < min_degree:
                    min_node = node
        else:
            print("here3")

        return min_node

    def find_path_exterior_nodes(self, path):
        import networkx as nx  # Ensure NetworkX is imported if it's used.
        G = nx.Graph(self.graph)
        last_node = path[-1]
        path_length = len(path)

        ext_neighbors = set()
        for neighbor in G.neighbors(last_node):
            if self.distance(path[0], neighbor) == path_length + 1:
                ext_neighbors.add(neighbor)

        return list(ext_neighbors)
		
    def find_path_neighbors(self, path):
        if len(path) < 2:
            raise ValueError("Path should have at least two nodes.")

        v0 = path[0]
        vk = path[-1]
        k = len(path) - 1
        distances = self.bfs_distance(v0)

        path_external_neighbors = {x for x in range(len(self.graph)) if self.graph[vk][x] == 1 and distances[x] == k + 1}
        path_internal_neighbors = {x for x in range(len(self.graph)) if self.graph[vk][x] == 1 and distances[x] <= k}

        return path_external_neighbors, path_internal_neighbors

    def find_seymour_diamonds(self):
        num_nodes = len(self.graph)
        diamonds = []

        for x in range(num_nodes):
            N_plus_x = self.first_degree_neighbors(x)

            for i in range(len(N_plus_x)):
                for j in range(i + 1, len(N_plus_x)):
                    u = N_plus_x[i]
                    v = N_plus_x[j]
                    if u in self.graph and v in self.graph[u]:
                        for w in range(num_nodes):
                            if (w != x and w in self.graph[u] and w in self.graph[v]):
                                diamonds.append((x, u, v, w))
        return diamonds

    def find_Seymour_graph(self, path):
        v0 = path[0]
        vk = path[-1]

        path_external_neighbors = self.find_path_exterior_nodes(path)

        S = set([v0, vk])
        for v in path_external_neighbors:
            S.add(v)

        return S

    def first_degree_neighbors(self, node):
        if not isinstance(self.graph, list) or not 0 <= node < len(self.graph):
            raise ValueError("Invalid node or graph.")

        neighbors = [i for i, is_connected in enumerate(self.graph[node]) if is_connected != 0]
        return neighbors

		
    def has_decreasing_sequence_property(self, node):
        N_plus_x = self.first_degree_neighbors(node)
        N_plus_plus_x = self.second_degree_neighbors(node)
        return len(N_plus_x) > len(N_plus_plus_x)

    def int_w_x(self, node, w):
        N_plus_w = self.first_degree_neighbors(w)
        int_neighbors = [x for x in N_plus_w if node in self.first_degree_neighbors(x)]
        return int_neighbors

    def interior_neighbors(self, y):
        N_plus_y = self.first_degree_neighbors(y)
        return [neighbor for neighbor in N_plus_y if neighbor not in self.second_degree_neighbors(y)]
	
    def second_degree_neighbors(self, node):
        first_neighbors = self.first_degree_neighbors(node)
        second_neighbors = set()
        for neighbor in first_neighbors:
            second_neighbors.update(self.first_degree_neighbors(neighbor))
        return list(second_neighbors - set(first_neighbors))

    def third_degree_neighbors(self, node):
        second_neighbors = self.second_degree_neighbors(node)
        third_neighbors = set()
        for neighbor in second_neighbors:
            third_neighbors.update(self.first_degree_neighbors(neighbor))
        return list(third_neighbors - set(second_neighbors))

    def validate_sequence_property(self):
        for node in self.graph:
            if not self.has_decreasing_sequence_property(node):
                return False
        return True
