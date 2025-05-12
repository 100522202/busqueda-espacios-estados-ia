# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic to implement, Manhattan """

    global NODES_EXPANDED

    # Obtenemos la fila y columna del nodo actual y del objetivo
    fila_actual = current_node[0]
    col_actual = current_node[1]
    fila_objetivo = objective_node[0]
    col_objetivo = objective_node[1]

    # Esta heurística calcula la distancia Manhattan,
    # que es la suma de los movimientos verticales y horizontales.
    distancia = abs(fila_actual - fila_objetivo) + abs(col_actual - col_objetivo)

    # Contamos la expansión del nodo 
    NODES_EXPANDED += 1

    return np.float32(distancia)


def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic to implement, Euclidea """

    global NODES_EXPANDED

    # Obtenemos coordenadas del nodo actual y del nodo objetivo
    y1 = current_node[0]
    x1 = current_node[1]
    y2 = objective_node[0]
    x2 = objective_node[1]

    # Esta heurística calcula la distancia euclidiana en línea recta,
    # como si usaras una regla entre los dos puntos.
    distancia = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

    NODES_EXPANDED += 1

    return np.float32(distancia)

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors

    alto = detection_map.shape[0]
    ancho = detection_map.shape[1]

    G = nx.DiGraph()  # Creamos un grafo dirigido vacío

    # Añadimos los nodos que tienen un valor de detección aceptable (menor o igual que el umbral)
    for fila in range(alto):
        for columna in range(ancho):
            valor = detection_map[fila, columna]
            if valor <= tolerance:
                G.add_node((fila, columna))

    # Posibles movimientos (solo en cruz)
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Añadimos las aristas entre celdas vecinas si ambas son válidas
    for fila in range(alto):
        for columna in range(ancho):
            valor = detection_map[fila, columna]
            if valor <= tolerance:
                for mov in movimientos:
                    nueva_fila = fila + mov[0]
                    nueva_columna = columna + mov[1]

                    if 0 <= nueva_fila < alto and 0 <= nueva_columna < ancho:
                        valor_vecino = detection_map[nueva_fila, nueva_columna]
                        if valor_vecino <= tolerance:
                            # El peso de la arista es el promedio entre origen y destino
                            peso = (valor + valor_vecino) / 2
                            G.add_edge((fila, columna), (nueva_fila, nueva_columna), weight=np.float32(peso))

    return G

def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordiantes from (lat, lon) into (x, y) """

    resultado = np.zeros_like(high_level_plan, dtype=np.int32)

    for i in range(len(high_level_plan)):
        latitud = high_level_plan[i][0]
        longitud = high_level_plan[i][1]

        # Normalizamos las coordenadas geográficas a una escala de 0 a mapa-1
        lon_total = boundaries.max_lon - boundaries.min_lon
        lat_total = boundaries.max_lat - boundaries.min_lat

        x_normalizado = (longitud - boundaries.min_lon) / lon_total
        y_normalizado = (latitud - boundaries.min_lat) / lat_total

        columna = int(x_normalizado * (map_width - 1))
        fila = int(y_normalizado * (map_height - 1))

        # Nos aseguramos de no salirnos del mapa
        columna = max(0, min(columna, map_width - 1))
        fila = max(0, min(fila, map_height - 1))

        resultado[i][0] = fila
        resultado[i][1] = columna

    return resultado

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array,
                 initial_location_index: np.int32,
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm, A* """

    global NODES_EXPANDED
    NODES_EXPANDED = 0

    # Convertimos los POIs en coordenadas de celda (índices del mapa)
    discretized = discretize_coords(locations, boundaries, map_width, map_height)

    # Reordenamos los POIs para que empiece desde el índice que se indique
    reorden = np.roll(discretized, -initial_location_index, axis=0)

    plan = []  # Aquí guardaremos las rutas encontradas

    for i in range(len(reorden) - 1):
        origen = tuple(map(int, reorden[i]))
        destino = tuple(map(int, reorden[i + 1]))

        if not G.has_node(origen) or not G.has_node(destino):
            print("No existe el nodo:", origen, "o", destino)
            continue

        try:
            # Buscamos la ruta óptima usando A*
            ruta = nx.astar_path(G, origen, destino, heuristic=heuristic_function)
        except nx.NetworkXNoPath:
            print("No hay camino entre:", origen, "y", destino)
            continue

        # Codificamos la ruta como texto tipo "(y, x)" para poder visualizarla después
        camino_codificado = []
        for punto in ruta:
            y = punto[0]
            x = punto[1]
            texto = f"({y}, {x})"
            camino_codificado.append(texto)

        plan.append(camino_codificado)

    return (plan, NODES_EXPANDED)

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """

    if not solution_plan:
        return np.float32(float('inf'))  # Si no hay ruta, el coste es infinito

    total = 0.0

    for segmento in solution_plan:
        for i in range(len(segmento) - 1):
            nodo1 = eval(segmento[i])  # Convertimos "(y, x)" a tupla
            nodo2 = eval(segmento[i + 1])

            if G.has_edge(nodo1, nodo2):
                peso = G[nodo1][nodo2]['weight']
                total += peso
            else:
                return np.float32(float('inf'))  # Si falta una arista, el camino no es válido

    return np.float32(total)
