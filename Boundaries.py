# Required imports
import numpy as np

class Boundaries:
    """ Class that defines the limits (in geodetic coordinates) of a map """
    def __init__(self, 
                 max_lat: np.float32, 
                 min_lat: np.float32,
                 max_lon: np.float32,
                 min_lon: np.float32):
        self.max_lat = max_lat      # Maximum latitude of the map
        self.min_lat = min_lat      # Minimum latitude of the map
        self.max_lon = max_lon      # Maximum longitude of the map
        self.min_lon = min_lon      # Minimum longitude of the map
