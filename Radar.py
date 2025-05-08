# Required imports
import numpy as np
from Location import Location

class Radar:
    """ Class that models the Radar """
    def __init__(self, 
                 location:           Location,
                 transmission_power: np.float32,
                 antenna_gain:       np.float32,
                 wavelength:         np.float32,
                 cross_section:      np.float32,
                 minimum_signal:     np.float32,
                 total_loss:         np.float32,
                 covariance:         np.array):
        self.location           = location              # Location of the radar (geodetic coordinates)
        self.transmission_power = transmission_power    # Transmission power (in Watts [W])
        self.antenna_gain       = antenna_gain          # Antenna gain (no units)
        self.wavelength         = wavelength            # Wavelength (in meters)
        self.cross_section      = cross_section         # Cross-section of the antenna (in squared meters)
        self.minimum_signal     = minimum_signal        # Sensitivity of the radar (in Watts [W])
        self.total_loss         = total_loss            # Loss of the radar (no units, discrete)

        # If the covariance matrix is NOT provided, then compute it
        if not covariance:
            self.covariance = self.get_covariance_matrix()
        else:
            self.covariance = covariance

    def get_covariance_matrix(self) -> np.array:
        """ Computes a random 2D-covariance matrix (ensuring semi-positive definite properties) """
        var_A, var_B = np.random.uniform(size=2, low=2e-5, high=2e-4)
        A = np.array([[ var_A, 0.0 ], [ 0.0, var_B ]])
        return A

    def compute_max_range(self) -> np.array:
        """ Computes the maximum detection range of the radar based on its operational properties """
        A = self.transmission_power * (self.antenna_gain ** 2) * (self.wavelength ** 2) * self.cross_section
        B = ((4.0 * np.pi) ** 3) * self.minimum_signal * self.total_loss
        return (A / B) ** (1  / 4)

    def compute_detection_level(self, latitude: np.float32, longitude: np.float32) -> np.float32:
        """ Computes the detection level of a given radar in a particular point in space """
        # Compute the radar's max range
        max_range = self.compute_max_range()

        # Compute an approximation of the distance (in meters) from the radar to the point
        distance = np.sqrt( (latitude - self.location.latitude) ** 2 + \
                            (longitude - self.location.longitude) ** 2 ) * 111000
        
        # If the distance is inside the detection range, proceed
        if distance <= max_range:
            # Compute a 2D-multivariate gaussian that models the attenuation of the radar's detections
            inv_cov_matrix = np.linalg.inv(a=self.covariance)
            det_cov_matrix = np.linalg.det(a=self.covariance)
            discrepancy    = np.array([ latitude, longitude ]) - self.location.to_numpy()   # x - mu
            A = 1.0 / (2.0 * np.pi * np.sqrt(det_cov_matrix))
            _exp = -0.5 * discrepancy @ inv_cov_matrix @ discrepancy
            return A * np.exp(_exp)
        
        # If not, return 0
        else:
            return 0.0
