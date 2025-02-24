
import random
import numpy as np
from models import *


#
# Add your Filtering / Smoothing approach(es) here
#

class HMMFilter:
    def __init__(self, initial_probs, transition_matrix, observation_matrix):
        self.__tm = transition_matrix  # Transition matrix (T)
        self.__om = observation_matrix  # Observation model (O)
        self.__f = initial_probs  # Initial belief distribution

    def filter(self, sensorR: int) -> np.array:
        """Perform HMM filtering given a sensor reading (index)."""
        
        # Step 1: Compute observation likelihood P(z | x)
        O_z = np.diag(self.__om[:, sensorR])  # Extract the observation model for this reading
        
        # Step 2: Predict next belief using transition model
        prediction = self.__tm.T @ self.__f  # f_prior = T^T * f
        
        # Step 3: Update belief using observation model
        updated_f = O_z @ prediction  # f_posterior = O * f_prior
        
        # Step 4: Normalize (ensures probabilities sum to 1)
        updated_f /= np.sum(updated_f)
        
        # Store updated belief for the next iteration
        self.__f = updated_f
        
        return updated_f



class HMMSmoother:
    def __init__(self, transition_matrix, observation_matrix):
        self.__tm = transition_matrix  # Transition matrix (T)
        self.__om = observation_matrix  # Observation model (O)

    def smooth(self, sensor_r_seq: np.array, f_k: np.array) -> np.array:
        """Perform HMM smoothing using the Forward-Backward algorithm."""
        num_states = len(f_k)
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
