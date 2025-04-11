# Required imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os
from Map import Map
from Boundaries import Boundaries
from SearchEngine import build_graph, path_finding, compute_path_cost, h1, h2

def plot_radar_locations(boundaries: Boundaries, radar_locations: np.array) -> None:
    """ Auxiliary function for plotting the radar locations """
    plt.figure(figsize=(8, 8))
    plt.title("Radar locations in the map")
    plt.plot([boundaries.min_lon, boundaries.max_lon, boundaries.max_lon, boundaries.min_lon, boundaries.min_lon],
             [boundaries.max_lat, boundaries.max_lat, boundaries.min_lat, boundaries.min_lat, boundaries.max_lat],
             label='Boundaries',
             linestyle='--',
             c='black')
    plt.scatter(radar_locations[:, 1], radar_locations[:, 0], label='Radars', c='green')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()
    return

def plot_detection_fields(detection_map: np.array, bicubic: bool=True) -> None:
    """ Auxiliary function for plotting the detection fields """
    plt.figure(figsize=(8, 8))
    plt.title("Radar detection fields")
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic' if bicubic else None)
    plt.colorbar(im, label='Detection values')
    plt.show()
    return

def plot_solution(detection_map: np.array, solution_plan: list, bicubic: bool=True) -> None:
    """ Auxiliary function for plotting the solution plan with markers in each POI """
    plt.figure(figsize=(8,8))
    plt.title("Solution plan")
    for i in range(len(solution_plan)):
        start_point = eval(solution_plan[i][0])
        plt.scatter(start_point[1], start_point[0], c='black', marker='*', zorder=2)
        path_array = np.zeros(shape=(len(solution_plan[i]), 2))
        for j in range(len(path_array)):
            path_array[j] = eval(solution_plan[i][j])
        plt.plot(path_array[:, 1], path_array[:, 0], zorder=1)
    final_point = eval(solution_plan[-1][-1])
    plt.scatter(final_point[1], final_point[0], c='black', marker='*', label=f'Waypoints', zorder=2)
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic' if bicubic else None)
    plt.colorbar(im, label='Detection values')
    plt.legend()
    plt.show()
    return

def parse_args() -> dict:
    """ Parses the main arguments of the program and returns them stored in a dictionary """
    json_path            = f"{os.getcwd()}/scenarios.json"
    scenario_json        = sys.argv[1]
    tolerance            = float(sys.argv[2])
    execution_parameters = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            key = list(entry.keys())[0]
            if key == scenario_json:
                execution_parameters = entry[key]
                break
    execution_parameters["tolerance"] = tolerance
    return execution_parameters

# System's main function
def main() -> None:

    # Parse the input parameters (arguments) of the program (current execution)
    execution_parameters = parse_args()

    # Set the pseudo-random number generator seed (DO NOT MODIFY)
    np.random.seed(42)

    # Set boundaries
    boundaries = Boundaries(max_lat=execution_parameters['max_lat'],
                            min_lat=execution_parameters['min_lat'],
                            max_lon=execution_parameters['max_lon'],
                            min_lon=execution_parameters['min_lon'])
    
    # Define the map with its corresponding boundaries and coordinates
    M = Map(boundaries=boundaries,
            height=execution_parameters['H'],
            width=execution_parameters['W'])
    
    # Generate random radars
    n_radars = execution_parameters['n_radars']
    M.generate_radars(n_radars=n_radars)
    radar_locations = M.get_radars_locations_numpy()

    # Plot the radar locations (latitude increments from bottom to top)
    plot_radar_locations(boundaries=boundaries, radar_locations=radar_locations)

    # Compute the detection map (sets the costs for each cell)
    detection_map = M.compute_detection_map()

    # Plot the detection map (detection fields)
    plot_detection_fields(detection_map=detection_map)

    # Build the graph from the detection map
    G = build_graph(detection_map=detection_map, tolerance=execution_parameters['tolerance'])

    # Get the POI's that the plane must visit
    POIs = np.array(execution_parameters['POIs'], dtype=np.float32)

    # Compute the solution
    solution_plan, nodes_expanded = path_finding(G=G,
                                 heuristic_function=h2,
                                 locations=POIs, 
                                 initial_location_index=0,
                                 boundaries=boundaries,
                                 map_width=M.width,
                                 map_height=M.height)
    
    # Compute the solution cost
    path_cost = compute_path_cost(G=G, solution_plan=solution_plan)

    # Some verbose of the total cost and the number of expanded nodes
    print(f"Total path cost: {path_cost}")
    print(f"Number of expanded nodes: {nodes_expanded}")

    # Plot the solution
    plot_solution(detection_map=detection_map, solution_plan=solution_plan)

if __name__ == '__main__':
    main()
