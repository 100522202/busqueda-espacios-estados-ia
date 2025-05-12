# Required imports
import numpy as np
from Location import Location
from Boundaries import Boundaries
from Radar import Radar
from tqdm import tqdm

# Constant that avoids setting cells to have an associated cost of zero
EPSILON = 1e-4

class Map:
    """ Class that models the map for the simulation """
    def __init__(self, 
                 boundaries: Boundaries,
                 height:     np.int32, 
                 width:      np.int32, 
                 radars:     np.array=None):
        self.boundaries = boundaries        # Boundaries of the map
        self.height     = height            # Number of coordinates in the y-axis
        self.width      = width             # Number of coordinates int the x-axis
        self.radars     = radars            # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        rand_lats = np.random.choice(a=lat_range, size=n_radars, replace=False)
        rand_lons = np.random.choice(a=lon_range, size=n_radars, replace=False)
        self.radars = []        # Initialize 'radars' as an empty list

        # Loop for each radar that must be generated
        for i in range(n_radars):
            # Create a new radar with random parameters
            new_radar = Radar(location=Location(latitude=rand_lats[i], longitude=rand_lons[i]),
                              transmission_power=np.random.uniform(low=1, high=1000000),
                              antenna_gain=np.random.uniform(low=10, high=50),
                              wavelength=np.random.uniform(low=0.001, high=10.0),
                              cross_section=np.random.uniform(low=0.1, high=10.0),
                              minimum_signal=np.random.uniform(low=1e-10, high=1e-15),
                              total_loss=np.random.randint(low=1, high=10),
                              covariance=None)

            # Insert the new radar
            self.radars.append(new_radar)
        return
    
    def get_radars_locations_numpy(self) -> np.array:
        """ Returns an array with the coordiantes (lat, lon) of each radar registered in the map """
        locations = np.zeros(shape=(len(self.radars), 2), dtype=np.float32)

        # Recorremos cada radar y guardamos su latitud y longitud en el array
        for i in range(len(self.radars)):
            locations[i] = self.radars[i].location.to_numpy()
        return locations
    
    def compute_detection_map(self) -> np.array:
        """ Computes the detection map for each coordinate in the map (with all the radars) """

        # Creamos una malla de latitudes y longitudes que cubre el mapa entero
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        
        # Creamos la matriz donde se guardará la detección de cada celda
        detection_map = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        
        # Para cada celda del mapa, calculamos cuánto se detecta allí
        for i in tqdm(range(self.height), desc="Computing detection map"):
            for j in range(self.width):
                # Coordenadas reales de esta celda
                lat = lat_range[i]
                lon = lon_range[j]
                
                # Calculamos la mayor detección recibida desde todos los radares
                max_detection = 0.0
                for radar in self.radars:
                    detection_level = radar.compute_detection_level(latitude=lat, longitude=lon)
                    max_detection = max(max_detection, detection_level)
                
                # Guardamos ese valor máximo en la celda correspondiente
                detection_map[i, j] = max_detection
        
        # Normalizamos todos los valores entre 0 y 1
        if np.max(detection_map) > 0:
            min_value = np.min(detection_map)
            max_value = np.max(detection_map)
            if max_value > min_value:  # Evitamos dividir por cero
                detection_map = (detection_map - min_value) / (max_value - min_value)
            
            # Añadimos EPSILON a las celdas muy bajas para evitar valores 0 en los costes
            detection_map = np.where(detection_map < EPSILON, EPSILON, detection_map)
        
        return detection_map
