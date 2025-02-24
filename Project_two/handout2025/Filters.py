
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__f = probs
        
    # sensorR is the sensor reading (index!), self._f is the probability distribution resulting from the filtering    
    def filter(self, sensorR: int) -> np.array:
        """
        Perform HMM filtering given a sensor reading.

        :param sensorR: Index of sensor reading
        :return: Updated belief state (probability distribution)
        """
        # Extract observation likelihood for the given sensor reading
        O_z = np.diag(self.__om[:, sensorR])  # Extract observation model

        # Predict next state belief using transition model
        prediction = self.__tm.T @ self.__f  # f_prior = T^T * f

        # Update belief using observation model
        updated_f = O_z @ prediction  # f_posterior = O * f_prior

        # Normalize (ensures probabilities sum to 1)
        updated_f /= np.sum(updated_f)

        # Store updated belief for the next step
        self.__f = updated_f
        
        return self.__f

class HMMSmoother:
    
    def __init__(self, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm

    # sensor_r_seq is the sequence (array) with the t-k sensor readings for smoothing, 
    # f_k is the filtered result (f_vector) for step k
    # fb is the smoothed result (fb_vector)

    def smooth(self, sensor_r_seq: np.array, f_k: np.array) -> np.array:
        """
        Perform HMM smoothing using the Forward-Backward algorithm.

        :param sensor_r_seq: Array of past sensor readings
        :param f_k: Filtered probability distributions from HMMFilter
        :return: Smoothed probability distributions
        """
        num_states = len(f_k[0])
        num_timesteps = len(sensor_r_seq)

        # Initialize backward message b_k with all 1s
        b_k = np.ones(num_states)

        # Smoothed belief initialization
        fb = np.zeros((num_timesteps, num_states))

        # Iterate backward through time
        for t in range(num_timesteps - 1, -1, -1):
            # Compute smoothed estimate: fb_t = α * f_t * b_t
            fb[t] = f_k[t] * b_k
            fb[t] /= np.sum(fb[t])  # Normalize

            # Compute new backward message
            if t > 0:
                O_next = np.diag(self.__om[:, sensor_r_seq[t]])  # Observation matrix for next reading
                b_k = self.__tm @ O_next @ b_k  # Backward step

                # Normalize
                b_k /= np.sum(b_k)

        return fb