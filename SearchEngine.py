# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON
import heapq

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic: Euclidean distance (admissible) """
    global NODES_EXPANDED
    # Parse node coordinates from string format "(y, x)"
    current = eval(current_node)
    objective = eval(objective_node)
    
    # Calculate Euclidean distance between current position and objective
    h = np.sqrt((current[0] - objective[0])**2 + (current[1] - objective[1])**2)
    
    NODES_EXPANDED += 1
    return h

def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic: Manhattan distance (admissible) * EPSILON
    This works better than h1 when navigating through grid-based maps with radar fields,
    as it's still admissible but provides better estimates than simple Euclidean distance
    """
    global NODES_EXPANDED
    # Parse node coordinates from string format "(y, x)"
    current = eval(current_node)
    objective = eval(objective_node)
    
    # Calculate Manhattan distance between current position and objective
    # Multiplied by EPSILON to ensure admissibility (since detection costs are at least EPSILON)
    h = (abs(current[0] - objective[0]) + abs(current[1] - objective[1])) * EPSILON
    
    NODES_EXPANDED += 1
    return h

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Get dimensions of the detection map
    height, width = detection_map.shape
    
    # Four possible movement directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Iterate through all cells in the grid
    for i in range(height):
        for j in range(width):
            # Create node for current cell
            current_node = f"({i}, {j})"
            G.add_node(current_node)
            
            # Check adjacent cells (neighbors)
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                # Check if neighbor is within grid boundaries
                if 0 <= ni < height and 0 <= nj < width:
                    # Create node for neighbor
                    neighbor_node = f"({ni}, {nj})"
                    
                    # Calculate edge weight based on detection probability at the neighbor
                    # If detection value > tolerance, we want to avoid this cell (high cost)
                    detection_value = detection_map[ni, nj]
                    
                    # Add edge from current node to neighbor with weight based on detection value
                    if detection_value <= tolerance:
                        # If below tolerance, assign minimum cost (EPSILON)
                        edge_weight = EPSILON
                    else:
                        # If above tolerance, assign cost based on detection value
                        edge_weight = detection_value
                    
                    # Add the edge to the graph
                    G.add_edge(current_node, neighbor_node, weight=edge_weight)
    
    return G

def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordiantes from (lat, lon) into (i, j) grid indices """
    # Calculate the step size for each dimension
    lat_step = (boundaries.max_lat - boundaries.min_lat) / (map_height - 1)
    lon_step = (boundaries.max_lon - boundaries.min_lon) / (map_width - 1)
    
    # Initialize the output array
    discretized_coords = np.zeros(shape=(high_level_plan.shape[0], 2), dtype=np.int32)
    
    # Convert each latitude and longitude coordinate to grid indices
    for i in range(high_level_plan.shape[0]):
        # Calculate i-index (row) - latitude
        discretized_coords[i, 0] = int(round((high_level_plan[i, 0] - boundaries.min_lat) / lat_step))
        
        # Calculate j-index (column) - longitude
        discretized_coords[i, 1] = int(round((high_level_plan[i, 1] - boundaries.min_lon) / lon_step))
        
        # Ensure the indices are within valid range
        discretized_coords[i, 0] = max(0, min(map_height - 1, discretized_coords[i, 0]))
        discretized_coords[i, 1] = max(0, min(map_width - 1, discretized_coords[i, 1]))
    
    return discretized_coords

def a_star_search(G: nx.DiGraph, start_node: str, goal_node: str, heuristic_function) -> list:
    """
    Implements A* search algorithm between two points in the graph
    Returns the shortest path from start to goal
    """
    # Priority queue for frontier nodes (f_score, node)
    frontier = [(0, start_node)]
    
    # Set of visited nodes
    visited = set()
    
    # Dictionary to store the cost to reach each node from the start
    g_score = {start_node: 0}
    
    # Dictionary to store the parent of each node in the optimal path
    parent = {}
    
    while frontier:
        # Get the node with lowest f_score
        _, current = heapq.heappop(frontier)
        
        # If we reached the goal, reconstruct and return the path
        if current == goal_node:
            path = [current]
            while path[0] != start_node:
                path.insert(0, parent[path[0]])
            return path
        
        # Skip if already visited
        if current in visited:
            continue
        
        # Mark as visited
        visited.add(current)
        
        # Explore neighbors
        for neighbor in G.neighbors(current):
            # Calculate tentative g_score for this neighbor
            tentative_g = g_score[current] + G[current][neighbor]['weight']
            
            # If we found a better path to this neighbor
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # Record this better path
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Calculate f_score = g_score + heuristic
                f_score = tentative_g + heuristic_function(neighbor, goal_node)
                
                # Add to frontier with priority based on f_score
                heapq.heappush(frontier, (f_score, neighbor))
    
    # If we get here, no path was found
    return []

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm """
    global NODES_EXPANDED
    NODES_EXPANDED = 0  # Reset counter
    
    # Discretize the high-level locations to grid coordinates
    discretized_locations = discretize_coords(locations, boundaries, map_width, map_height)
    
    # Create list to hold all paths in the solution
    solution_plan = []
    
    # Define the visit order starting from the initial location index
    num_locations = len(locations)
    visit_order = [(initial_location_index, (i+1+initial_location_index) % num_locations) 
                  for i in range(num_locations-1)]
    
    # For each pair of consecutive locations to visit
    for from_idx, to_idx in visit_order:
        # Get grid coordinates for the current locations
        from_pos = discretized_locations[from_idx]
        to_pos = discretized_locations[to_idx]
        
        # Convert to node format for the graph
        from_node = f"({from_pos[0]}, {from_pos[1]})"
        to_node = f"({to_pos[0]}, {to_pos[1]})"
        
        # Find path between these locations using A* search
        path = a_star_search(G, from_node, to_node, heuristic_function)
        
        # Add the path to the solution plan
        if path:
            solution_plan.append(path)
        else:
            print(f"Warning: No path found between {from_node} and {to_node}")
    
    return solution_plan, NODES_EXPANDED

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    total_cost = 0.0
    
    # Iterate through each path segment in the solution plan
    for path in solution_plan:
        # Calculate the cost of this path segment
        for i in range(len(path) - 1):
            # Add the edge weight (cost) between consecutive nodes
            total_cost += G[path[i]][path[i+1]]['weight']
    
    return total_cost