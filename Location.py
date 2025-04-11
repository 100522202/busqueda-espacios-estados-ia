# Required imports
import numpy as np

class Location:
    """ Class that represents a bidimensional (geodetic) coordinate / location """
    def __init__(self, latitude: np.float32, longitude: np.float32):
        self.latitude  = latitude       # Latitude of the object
        self.longitude = longitude      # Longitude of the object

    def to_numpy(self) -> np.array:
        """ Converts the current Location object into a 2-element Numpy array """
        return np.array([ self.latitude, self.longitude ])
