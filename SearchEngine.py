import numpy as np
import networkx as nx
import heapq
from typing import List, Tuple, Optional, Callable
from Boundaries import Boundaries
from Map import EPSILON

NODES_EXPANDED = 0

def h1(current_node: Tuple[int, int], objective_node: Tuple[int, int]) -> np.float32:
    global NODES_EXPANDED
    h = abs(current_node[0] - objective_node[0]) + abs(current_node[1] - objective_node[1])
    NODES_EXPANDED += 1
    return np.float32(h)

def h2(current_node: Tuple[int, int], objective_node: Tuple[int, int]) -> np.float32:
    global NODES_EXPANDED
    h = np.sqrt((current_node[0] - objective_node[0]) ** 2 + (current_node[1] - objective_node[1]) ** 2)
    NODES_EXPANDED += 1
    return np.float32(h)

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    height, width = detection_map.shape
    G = nx.DiGraph()

    for i in range(height):
        for j in range(width):
            if detection_map[i, j] <= tolerance:
                G.add_node((i, j))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(height):
        for j in range(width):
            if detection_map[i, j] <= tolerance:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < height and 0 <= nj < width and detection_map[ni, nj] <= tolerance):
                        cost = np.float32((detection_map[i, j] + detection_map[ni, nj]) / 2)
                        G.add_edge((i, j), (ni, nj), weight=cost)

    return G

def discretize_coords(high_level_plan: np.array,
                      boundaries: Boundaries,
                      map_width: int,
                      map_height: int) -> np.array:
    discretized_plan = np.zeros_like(high_level_plan, dtype=np.int32)

    for i in range(len(high_level_plan)):
        lat, lon = high_level_plan[i]
        x_norm = (lon - boundaries.min_lon) / (boundaries.max_lon - boundaries.min_lon)
        y_norm = (lat - boundaries.min_lat) / (boundaries.max_lat - boundaries.min_lat)
        x = int(x_norm * (map_width - 1))
        y = int(y_norm * (map_height - 1))
        x = max(0, min(x, map_width - 1))
        y = max(0, min(y, map_height - 1))
        discretized_plan[i] = [y, x]

    return discretized_plan

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

    # Step 1: Discretize coordinates (lat, lon) → (i, j)
    discrete_locations = discretize_coords(locations, boundaries, map_width, map_height)

    solution_plan = []

    # Step 2: Apply A* between each pair of POIs
    for i in range(len(discrete_locations) - 1):
        start = tuple(discrete_locations[i])
        goal = tuple(discrete_locations[i + 1])

        try:
            # Run A*
            path = nx.astar_path(G, start, goal, heuristic=heuristic_function)

            # Convert (i, j) to string '(y, x)'
            converted_path = []
            for y, x in path:
                converted_path.append(f"({y}, {x})")

            solution_plan.append(converted_path)

        except nx.NetworkXNoPath:
            print(f"❌ No hay camino entre {start} y {goal}")
            continue

    return solution_plan, NODES_EXPANDED

def compute_path_cost(G: nx.DiGraph, solution_plan: List[List[str]]) -> np.float32:
    if not solution_plan:
        return np.float32(float('inf'))

    total_cost = 0.0
    for path in solution_plan:
        for i in range(len(path) - 1):
            node_a = eval(path[i])
            node_b = eval(path[i + 1])
            y1, x1 = node_a
            y2, x2 = node_b
            if G.has_edge((y1, x1), (y2, x2)):
                total_cost += G[(y1, x1)][(y2, x2)]['weight']
            else:
                return np.float32(float('inf'))

    return np.float32(total_cost)
